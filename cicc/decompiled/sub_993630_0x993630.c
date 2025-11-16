// Function: sub_993630
// Address: 0x993630
//
char __fastcall sub_993630(__int64 a1, __int64 a2, _QWORD *a3, __int64 *a4, char a5)
{
  __int64 v5; // r10
  unsigned int v8; // r13d
  __int64 v9; // r14
  unsigned int v10; // eax
  char *v11; // rcx
  unsigned int v12; // r12d
  unsigned int v13; // r14d
  bool v14; // cc
  unsigned int v15; // eax
  __int64 v16; // rdi
  char result; // al
  unsigned int v18; // eax
  unsigned int v19; // r15d
  unsigned int v20; // r13d
  char *v21; // [rsp+8h] [rbp-B8h]
  __int64 v22; // [rsp+10h] [rbp-B0h] BYREF
  unsigned int v23; // [rsp+18h] [rbp-A8h]
  __int64 v24; // [rsp+20h] [rbp-A0h] BYREF
  unsigned int v25; // [rsp+28h] [rbp-98h]
  __int64 v26; // [rsp+30h] [rbp-90h] BYREF
  unsigned int v27; // [rsp+38h] [rbp-88h]
  __int64 v28; // [rsp+40h] [rbp-80h] BYREF
  unsigned int v29; // [rsp+48h] [rbp-78h]
  __int64 v30; // [rsp+50h] [rbp-70h] BYREF
  unsigned int v31; // [rsp+58h] [rbp-68h]
  __int64 v32; // [rsp+60h] [rbp-60h]
  unsigned int v33; // [rsp+68h] [rbp-58h]
  __int64 v34; // [rsp+70h] [rbp-50h] BYREF
  unsigned int v35; // [rsp+78h] [rbp-48h]
  __int64 v36; // [rsp+80h] [rbp-40h]
  int v37; // [rsp+88h] [rbp-38h]

  v5 = a1;
  v8 = *(_WORD *)(a2 + 2) & 0x3F;
  if ( a5 )
  {
    v18 = sub_B52870(v8);
    v5 = a1;
    v8 = v18;
  }
  v9 = *(_QWORD *)(a2 - 64);
  if ( *(_BYTE *)v9 != 67 || v5 != *(_QWORD *)(v9 - 32) )
    return sub_991B20(v5, v8, *(unsigned __int8 **)(a2 - 64), *(char **)(a2 - 32), (__int64)a3, a4);
  v21 = *(char **)(a2 - 32);
  v10 = sub_BCB060(*(_QWORD *)(v9 + 8));
  v11 = v21;
  v27 = v10;
  v12 = v10;
  if ( v10 > 0x40 )
  {
    sub_C43690(&v26, 0, 0);
    v29 = v12;
    sub_C43690(&v28, 0, 0);
    v11 = v21;
  }
  else
  {
    v29 = v10;
    v26 = 0;
    v28 = 0;
  }
  sub_991B20(v9, v8, (unsigned __int8 *)v9, v11, (__int64)&v26, a4);
  if ( (*(_BYTE *)(v9 + 1) & 2) != 0 )
  {
    v19 = *((_DWORD *)a3 + 2);
    v20 = v27;
    sub_C449B0(&v22, &v26, v19);
    if ( v20 != v23 )
    {
      if ( v20 > 0x3F || v23 > 0x40 )
        sub_C43C90(&v22, v20, v23);
      else
        v22 |= 0xFFFFFFFFFFFFFFFFLL >> ((unsigned __int8)v20 + 64 - (unsigned __int8)v23) << v20;
    }
    sub_C449B0(&v24, &v28, v19);
    v35 = v23;
    if ( v23 > 0x40 )
    {
      sub_C43780(&v34, &v22);
      v31 = v35;
      v30 = v34;
      v33 = v25;
      v32 = v24;
      if ( v23 > 0x40 && v22 )
        j_j___libc_free_0_0(v22);
    }
    else
    {
      v31 = v23;
      v30 = v22;
      v33 = v25;
      v32 = v24;
    }
    sub_987D70((__int64)&v34, a3, &v30);
    if ( *((_DWORD *)a3 + 2) <= 0x40u )
      goto LABEL_11;
    goto LABEL_9;
  }
  v13 = *((_DWORD *)a3 + 2);
  sub_C449B0(&v34, &v28, v13);
  sub_C449B0(&v24, &v26, v13);
  v31 = v25;
  v30 = v24;
  v33 = v35;
  v32 = v34;
  sub_987D70((__int64)&v34, a3, &v30);
  if ( *((_DWORD *)a3 + 2) > 0x40u )
  {
LABEL_9:
    if ( *a3 )
      j_j___libc_free_0_0(*a3);
  }
LABEL_11:
  v14 = *((_DWORD *)a3 + 6) <= 0x40u;
  *a3 = v34;
  v15 = v35;
  v35 = 0;
  *((_DWORD *)a3 + 2) = v15;
  if ( v14 || (v16 = a3[2]) == 0 )
  {
    a3[2] = v36;
    result = v37;
    *((_DWORD *)a3 + 6) = v37;
  }
  else
  {
    j_j___libc_free_0_0(v16);
    v14 = v35 <= 0x40;
    a3[2] = v36;
    result = v37;
    *((_DWORD *)a3 + 6) = v37;
    if ( !v14 && v34 )
      result = j_j___libc_free_0_0(v34);
  }
  if ( v33 > 0x40 && v32 )
    result = j_j___libc_free_0_0(v32);
  if ( v31 > 0x40 && v30 )
    result = j_j___libc_free_0_0(v30);
  if ( v29 > 0x40 && v28 )
    result = j_j___libc_free_0_0(v28);
  if ( v27 > 0x40 )
  {
    if ( v26 )
      return j_j___libc_free_0_0(v26);
  }
  return result;
}
