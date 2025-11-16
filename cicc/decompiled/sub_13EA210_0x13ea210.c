// Function: sub_13EA210
// Address: 0x13ea210
//
int *__fastcall sub_13EA210(int *a1, __int64 a2, __int64 a3)
{
  unsigned int v3; // eax
  unsigned int v5; // edx
  unsigned int v6; // eax
  unsigned int v7; // eax
  unsigned int v8; // eax
  unsigned int v10; // edx
  __int64 v11; // rbx
  bool v12; // cc
  char v13; // al
  bool v14; // r15
  unsigned int v15; // edx
  __int64 v16; // r15
  char v17; // al
  bool v18; // bl
  unsigned int v19; // [rsp+Ch] [rbp-74h]
  unsigned int v20; // [rsp+Ch] [rbp-74h]
  __int64 v21; // [rsp+10h] [rbp-70h] BYREF
  unsigned int v22; // [rsp+18h] [rbp-68h]
  __int64 v23; // [rsp+20h] [rbp-60h]
  unsigned int v24; // [rsp+28h] [rbp-58h]
  __int64 v25; // [rsp+30h] [rbp-50h] BYREF
  unsigned int v26; // [rsp+38h] [rbp-48h]
  __int64 v27; // [rsp+40h] [rbp-40h]
  unsigned int v28; // [rsp+48h] [rbp-38h]

  v3 = *(_DWORD *)a2;
  if ( !*(_DWORD *)a2 )
    goto LABEL_23;
  v5 = *(_DWORD *)a3;
  if ( !v5 || v3 == 4 )
    goto LABEL_25;
  if ( v5 == 4 )
    goto LABEL_23;
  if ( v3 == 3 )
  {
    v26 = *(_DWORD *)(a2 + 16);
    if ( v26 > 0x40 )
      sub_16A4FD0(&v25, a2 + 8);
    else
      v25 = *(_QWORD *)(a2 + 8);
    sub_16A7490(&v25, 1);
    v10 = v26;
    v11 = v25;
    v26 = 0;
    v12 = *(_DWORD *)(a2 + 32) <= 0x40u;
    v22 = v10;
    v21 = v25;
    if ( v12 )
    {
      v14 = *(_QWORD *)(a2 + 24) == v25;
    }
    else
    {
      v19 = v10;
      v13 = sub_16A5220(a2 + 24, &v21);
      v10 = v19;
      v14 = v13;
    }
    if ( v10 > 0x40 )
    {
      if ( v11 )
      {
        j_j___libc_free_0_0(v11);
        if ( v26 > 0x40 )
        {
          if ( v25 )
            j_j___libc_free_0_0(v25);
        }
      }
    }
    if ( v14 )
      goto LABEL_23;
    v3 = *(_DWORD *)a2;
  }
  if ( v3 == 1 )
  {
LABEL_23:
    *a1 = 0;
    sub_13E8810(a1, (unsigned int *)a2);
    return a1;
  }
  v6 = *(_DWORD *)a3;
  if ( *(_DWORD *)a3 == 3 )
  {
    v26 = *(_DWORD *)(a3 + 16);
    if ( v26 > 0x40 )
      sub_16A4FD0(&v25, a3 + 8);
    else
      v25 = *(_QWORD *)(a3 + 8);
    sub_16A7490(&v25, 1);
    v15 = v26;
    v16 = v25;
    v26 = 0;
    v12 = *(_DWORD *)(a3 + 32) <= 0x40u;
    v22 = v15;
    v21 = v25;
    if ( v12 )
    {
      v18 = *(_QWORD *)(a3 + 24) == v25;
    }
    else
    {
      v20 = v15;
      v17 = sub_16A5220(a3 + 24, &v21);
      v15 = v20;
      v18 = v17;
    }
    if ( v15 > 0x40 )
    {
      if ( v16 )
      {
        j_j___libc_free_0_0(v16);
        if ( v26 > 0x40 )
        {
          if ( v25 )
            j_j___libc_free_0_0(v25);
        }
      }
    }
    if ( v18 )
      goto LABEL_25;
    v6 = *(_DWORD *)a3;
  }
  if ( v6 == 1 )
  {
LABEL_25:
    *a1 = 0;
    sub_13E8810(a1, (unsigned int *)a3);
    return a1;
  }
  if ( *(_DWORD *)a2 != 3 || v6 != 3 )
    goto LABEL_23;
  sub_158BE00(&v21, a2 + 8, a3 + 8);
  v7 = v22;
  v22 = 0;
  v26 = v7;
  v25 = v21;
  v8 = v24;
  v24 = 0;
  v28 = v8;
  v27 = v23;
  sub_13EA060(a1, &v25);
  if ( v28 > 0x40 && v27 )
    j_j___libc_free_0_0(v27);
  if ( v26 > 0x40 && v25 )
    j_j___libc_free_0_0(v25);
  if ( v24 > 0x40 && v23 )
    j_j___libc_free_0_0(v23);
  if ( v22 > 0x40 && v21 )
    j_j___libc_free_0_0(v21);
  return a1;
}
