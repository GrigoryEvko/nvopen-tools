// Function: sub_158A180
// Address: 0x158a180
//
char __fastcall sub_158A180(__int64 a1, int *a2, __int64 a3)
{
  char result; // al
  unsigned int v6; // edx
  __int64 v7; // r8
  bool v8; // cc
  unsigned int v9; // edx
  __int64 v10; // r14
  unsigned int v11; // r14d
  __int64 v12; // rsi
  unsigned int v13; // ecx
  bool v14; // al
  unsigned int v15; // r14d
  __int64 v16; // rsi
  unsigned int v17; // ecx
  int v18; // eax
  int v19; // eax
  __int64 v20; // rsi
  __int64 v21; // rdx
  unsigned __int64 v22; // rdi
  __int64 v23; // rdx
  __int64 v24; // rdx
  __int64 v25; // rax
  unsigned __int64 v26; // rsi
  __int64 v27; // rax
  __int64 v28; // rdx
  __int64 v29; // rax
  unsigned __int64 v30; // rsi
  __int64 v31; // rax
  __int64 v32; // [rsp+0h] [rbp-60h]
  unsigned int v33; // [rsp+Ch] [rbp-54h]
  char v34; // [rsp+Ch] [rbp-54h]
  unsigned int v35; // [rsp+Ch] [rbp-54h]
  char v36; // [rsp+Ch] [rbp-54h]
  char v37; // [rsp+Ch] [rbp-54h]
  char v38; // [rsp+Ch] [rbp-54h]
  __int64 v39; // [rsp+10h] [rbp-50h] BYREF
  unsigned int v40; // [rsp+18h] [rbp-48h]
  __int64 v41; // [rsp+20h] [rbp-40h] BYREF
  unsigned int v42; // [rsp+28h] [rbp-38h]

  if ( sub_158A0B0(a1) || sub_158A120(a1) )
  {
    *a2 = 35 - ((sub_158A120(a1) == 0) - 1);
    v42 = *(_DWORD *)(a1 + 8);
    if ( v42 <= 0x40 )
      v41 = 0;
    else
      sub_16A4EF0(&v41, 0, 0);
    if ( *(_DWORD *)(a3 + 8) > 0x40u )
    {
      if ( *(_QWORD *)a3 )
        j_j___libc_free_0_0(*(_QWORD *)a3);
    }
    *(_QWORD *)a3 = v41;
    *(_DWORD *)(a3 + 8) = v42;
    return 1;
  }
  v40 = *(_DWORD *)(a1 + 8);
  if ( v40 > 0x40 )
    sub_16A4FD0(&v39, a1);
  else
    v39 = *(_QWORD *)a1;
  sub_16A7490(&v39, 1);
  v6 = v40;
  v7 = v39;
  v40 = 0;
  v8 = *(_DWORD *)(a1 + 24) <= 0x40u;
  v42 = v6;
  v41 = v39;
  if ( v8 )
  {
    result = *(_QWORD *)(a1 + 16) == v39;
  }
  else
  {
    v32 = v39;
    v33 = v6;
    result = sub_16A5220(a1 + 16, &v41);
    v7 = v32;
    v6 = v33;
  }
  if ( v6 > 0x40 )
  {
    if ( v7 )
    {
      v34 = result;
      j_j___libc_free_0_0(v7);
      result = v34;
      if ( v40 > 0x40 )
      {
        if ( v39 )
        {
          j_j___libc_free_0_0(v39);
          result = v34;
        }
      }
    }
  }
  if ( result )
  {
    *a2 = 32;
    if ( *(_DWORD *)(a3 + 8) > 0x40u || *(_DWORD *)(a1 + 8) > 0x40u )
    {
      v37 = result;
      sub_16A51C0(a3, a1);
      return v37;
    }
    v20 = *(_QWORD *)a1;
    *(_QWORD *)a3 = *(_QWORD *)a1;
    v21 = *(unsigned int *)(a1 + 8);
LABEL_59:
    *(_DWORD *)(a3 + 8) = v21;
    v22 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v21;
    if ( (unsigned int)v21 > 0x40 )
    {
      v23 = (unsigned int)((unsigned __int64)(v21 + 63) >> 6) - 1;
      *(_QWORD *)(v20 + 8 * v23) &= v22;
    }
    else
    {
      *(_QWORD *)a3 = v22 & v20;
    }
    return result;
  }
  v40 = *(_DWORD *)(a1 + 24);
  if ( v40 > 0x40 )
    sub_16A4FD0(&v39, a1 + 16);
  else
    v39 = *(_QWORD *)(a1 + 16);
  sub_16A7490(&v39, 1);
  v9 = v40;
  v10 = v39;
  v40 = 0;
  v8 = *(_DWORD *)(a1 + 8) <= 0x40u;
  v42 = v9;
  v41 = v39;
  if ( v8 )
  {
    result = *(_QWORD *)a1 == v39;
  }
  else
  {
    v35 = v9;
    result = sub_16A5220(a1, &v41);
    v9 = v35;
  }
  if ( v9 > 0x40 )
  {
    if ( v10 )
    {
      v36 = result;
      j_j___libc_free_0_0(v10);
      result = v36;
      if ( v40 > 0x40 )
      {
        if ( v39 )
        {
          j_j___libc_free_0_0(v39);
          result = v36;
        }
      }
    }
  }
  if ( result )
  {
    *a2 = 33;
    if ( *(_DWORD *)(a3 + 8) > 0x40u || *(_DWORD *)(a1 + 24) > 0x40u )
    {
      v38 = result;
      sub_16A51C0(a3, a1 + 16);
      return v38;
    }
    v20 = *(_QWORD *)(a1 + 16);
    *(_QWORD *)a3 = v20;
    v21 = *(unsigned int *)(a1 + 24);
    goto LABEL_59;
  }
  v11 = *(_DWORD *)(a1 + 8);
  v12 = *(_QWORD *)a1;
  v13 = v11 - 1;
  if ( v11 <= 0x40 )
  {
    v19 = 40;
    if ( v12 == 1LL << v13 )
      goto LABEL_52;
    v14 = v12 == 0;
  }
  else
  {
    if ( (*(_QWORD *)(v12 + 8LL * (v13 >> 6)) & (1LL << v13)) != 0 && (unsigned int)sub_16A58A0(a1) == v13 )
    {
      v19 = 40;
      goto LABEL_52;
    }
    v14 = v11 == (unsigned int)sub_16A57B0(a1);
  }
  if ( !v14 )
  {
    v15 = *(_DWORD *)(a1 + 24);
    v16 = *(_QWORD *)(a1 + 16);
    v17 = v15 - 1;
    if ( v15 <= 0x40 )
    {
      v18 = 39;
      if ( v16 != 1LL << v17 )
      {
        result = v16 == 0;
        goto LABEL_40;
      }
    }
    else
    {
      if ( (*(_QWORD *)(v16 + 8LL * (v17 >> 6)) & (1LL << v17)) == 0 || (unsigned int)sub_16A58A0(a1 + 16) != v17 )
      {
        result = v15 == (unsigned int)sub_16A57B0(a1 + 16);
LABEL_40:
        if ( !result )
          return result;
        v18 = 35;
        goto LABEL_42;
      }
      v18 = 39;
    }
LABEL_42:
    *a2 = v18;
    if ( *(_DWORD *)(a3 + 8) <= 0x40u && *(_DWORD *)(a1 + 8) <= 0x40u )
    {
      v28 = *(_QWORD *)a1;
      *(_QWORD *)a3 = *(_QWORD *)a1;
      v29 = *(unsigned int *)(a1 + 8);
      *(_DWORD *)(a3 + 8) = v29;
      v30 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v29;
      if ( (unsigned int)v29 > 0x40 )
      {
        v31 = (unsigned int)((unsigned __int64)(v29 + 63) >> 6) - 1;
        *(_QWORD *)(v28 + 8 * v31) &= v30;
      }
      else
      {
        *(_QWORD *)a3 = v30 & v28;
      }
      return 1;
    }
    else
    {
      sub_16A51C0(a3, a1);
      return 1;
    }
  }
  v19 = 36;
LABEL_52:
  *a2 = v19;
  if ( *(_DWORD *)(a3 + 8) <= 0x40u && *(_DWORD *)(a1 + 24) <= 0x40u )
  {
    v24 = *(_QWORD *)(a1 + 16);
    *(_QWORD *)a3 = v24;
    v25 = *(unsigned int *)(a1 + 24);
    *(_DWORD *)(a3 + 8) = v25;
    v26 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v25;
    if ( (unsigned int)v25 > 0x40 )
    {
      v27 = (unsigned int)((unsigned __int64)(v25 + 63) >> 6) - 1;
      *(_QWORD *)(v24 + 8 * v27) &= v26;
    }
    else
    {
      *(_QWORD *)a3 = v26 & v24;
    }
    return 1;
  }
  else
  {
    sub_16A51C0(a3, a1 + 16);
    return 1;
  }
}
