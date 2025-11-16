// Function: sub_986C90
// Address: 0x986c90
//
__int64 __fastcall sub_986C90(char *a1, int a2, _QWORD *a3)
{
  char v4; // al
  unsigned int v5; // r14d
  __int64 v7; // r10
  char **v8; // rax
  char *v9; // r15
  char *v10; // rax
  int v11; // ebx
  unsigned int v12; // r13d
  __int64 v13; // rax
  unsigned int v14; // eax
  __int64 v15; // [rsp+8h] [rbp-78h]
  __int64 v16; // [rsp+10h] [rbp-70h]
  unsigned int v17; // [rsp+1Ch] [rbp-64h]
  unsigned int v18; // [rsp+1Ch] [rbp-64h]
  __int64 v19; // [rsp+20h] [rbp-60h] BYREF
  unsigned int v20; // [rsp+28h] [rbp-58h]
  __int64 v21; // [rsp+30h] [rbp-50h] BYREF
  unsigned int v22; // [rsp+38h] [rbp-48h]
  __int64 v23[8]; // [rsp+40h] [rbp-40h] BYREF

  v4 = *a1;
  if ( (unsigned __int8)*a1 <= 0x1Cu )
  {
    v5 = 0;
    if ( v4 != 5 || *((_WORD *)a1 + 1) != 34 )
      return v5;
  }
  else
  {
    v5 = 0;
    if ( v4 != 63 )
      return v5;
  }
  v5 = 0;
  if ( (*((_DWORD *)a1 + 1) & 0x7FFFFFF) != 2 )
    return v5;
  if ( **((_BYTE **)a1 - 4) > 0x15u )
    return v5;
  v7 = *((_QWORD *)a1 - 8);
  if ( *(_BYTE *)v7 != 84 || (*(_DWORD *)(v7 + 4) & 0x7FFFFFF) != 2 )
    return v5;
  v8 = *(char ***)(v7 - 8);
  v9 = *v8;
  v10 = v8[4];
  if ( v9 && a1 == v9 )
  {
    v9 = v10;
  }
  else
  {
    LOBYTE(v5) = a1 == v10 && v10 != 0;
    if ( !(_BYTE)v5 )
      return v5;
  }
  v16 = *((_QWORD *)a1 - 8);
  v17 = sub_AE43F0(*a3, *((_QWORD *)v9 + 1));
  sub_9691E0((__int64)&v19, v17, 0, 0, 0);
  v5 = 0;
  v15 = sub_BD45C0((_DWORD)v9, *a3, (unsigned int)&v19, 0, 0, 0, 0, 0);
  sub_9691E0((__int64)&v21, v17, 0, 0, 0);
  if ( v16 == sub_BD45C0((_DWORD)a1, *a3, (unsigned int)&v21, 0, 0, 0, 0, 0) )
  {
    sub_9691E0((__int64)v23, v17, 0, 0, 0);
    if ( v15 == sub_BD45C0(a2, *a3, (unsigned int)v23, 0, 0, 0, 0, 0) )
    {
      v11 = sub_C4C880(&v19, v23);
      v12 = v22 - 1;
      if ( v11 < 0 )
        goto LABEL_27;
      v13 = 1LL << v12;
      if ( v22 > 0x40 )
      {
        if ( (*(_QWORD *)(v21 + 8LL * (v12 >> 6)) & v13) == 0 )
        {
          v18 = v22;
          v5 = 1;
          if ( (unsigned int)sub_C444A0(&v21) != v18 )
            goto LABEL_22;
        }
      }
      else if ( (v13 & v21) == 0 )
      {
        v5 = 1;
        if ( v21 )
          goto LABEL_22;
      }
      v5 = 0;
      if ( !v11 )
      {
LABEL_27:
        LOBYTE(v14) = sub_986C60(&v21, v12);
        v5 = v14;
      }
    }
LABEL_22:
    sub_969240(v23);
  }
  if ( v22 > 0x40 && v21 )
    j_j___libc_free_0_0(v21);
  if ( v20 > 0x40 && v19 )
    j_j___libc_free_0_0(v19);
  return v5;
}
