// Function: sub_AD8F60
// Address: 0xad8f60
//
unsigned __int8 *__fastcall sub_AD8F60(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rbx
  __int64 v5; // r15
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 v9; // rax
  __int64 v10; // r12
  __int64 v11; // rdx
  unsigned __int8 *v12; // r13
  int v13; // edx
  int v14; // eax
  _QWORD *i; // rbx
  __int64 v17; // [rsp+0h] [rbp-60h] BYREF
  unsigned int v18; // [rsp+8h] [rbp-58h]
  __int64 v19; // [rsp+10h] [rbp-50h] BYREF
  _QWORD *v20; // [rsp+18h] [rbp-48h]

  v4 = a1;
  if ( (unsigned int)*(unsigned __int8 *)(a1 + 8) - 17 <= 1 )
    a1 = **(_QWORD **)(a1 + 16);
  v5 = sub_BCAC60(a1);
  v9 = sub_C33340(a1, a2, v6, v7, v8);
  v10 = v9;
  if ( !a3 )
  {
    if ( v5 == v9 )
    {
      sub_C3C500(&v19, v9, 0);
      v11 = (unsigned __int8)a2;
      if ( v10 != v19 )
        goto LABEL_13;
    }
    else
    {
      sub_C373C0(&v19, v5, 0);
      v11 = (unsigned __int8)a2;
      if ( v10 != v19 )
      {
LABEL_13:
        sub_C36070(&v19, 0, v11, 0);
        goto LABEL_14;
      }
    }
    sub_C3D480(&v19, 0, v11, 0);
    goto LABEL_14;
  }
  v17 = a3;
  v18 = 64;
  if ( v5 == v9 )
    sub_C3C500(&v19, v9, 0);
  else
    sub_C373C0(&v19, v5, 0);
  if ( v19 == v10 )
    sub_C3D480(&v19, 0, (unsigned __int8)a2, &v17);
  else
    sub_C36070(&v19, 0, (unsigned __int8)a2, &v17);
  if ( v18 > 0x40 && v17 )
    j_j___libc_free_0_0(v17);
LABEL_14:
  v12 = (unsigned __int8 *)sub_AC8EA0(*(__int64 **)v4, &v19);
  v13 = *(unsigned __int8 *)(v4 + 8);
  if ( (unsigned int)(v13 - 17) <= 1 )
  {
    v14 = *(_DWORD *)(v4 + 32);
    BYTE4(v17) = (_BYTE)v13 == 18;
    LODWORD(v17) = v14;
    v12 = (unsigned __int8 *)sub_AD5E10(v17, v12);
  }
  if ( v10 == v19 )
  {
    if ( v20 )
    {
      for ( i = &v20[3 * *(v20 - 1)]; v20 != i; sub_91D830(i) )
        i -= 3;
      j_j_j___libc_free_0_0(i - 1);
    }
  }
  else
  {
    sub_C338F0(&v19);
  }
  return v12;
}
