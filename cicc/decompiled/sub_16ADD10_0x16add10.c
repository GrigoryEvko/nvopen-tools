// Function: sub_16ADD10
// Address: 0x16add10
//
unsigned __int64 __fastcall sub_16ADD10(__int64 a1, __int64 a2, unsigned __int64 *a3, unsigned __int64 *a4)
{
  __int64 v6; // rbx
  unsigned __int64 v7; // r14
  unsigned int v8; // edx
  unsigned __int64 v9; // rbx
  size_t v10; // rdx
  void *v11; // rdi
  unsigned __int64 result; // rax
  unsigned __int64 v13; // rax
  unsigned __int64 v14; // rdx
  __int64 v15; // rax
  unsigned __int64 v16; // rsi
  unsigned __int64 v17; // r14
  __int64 v18; // r15
  unsigned __int64 v19; // r14
  unsigned int v20; // eax
  __int64 v21; // rax
  unsigned int v22; // ecx
  __int64 v23; // rbx
  unsigned int v24; // ecx
  __int64 v25; // rax
  unsigned __int64 v26; // [rsp+8h] [rbp-58h]
  int v27; // [rsp+14h] [rbp-4Ch]
  unsigned int v28; // [rsp+14h] [rbp-4Ch]
  unsigned __int64 v29; // [rsp+18h] [rbp-48h]
  unsigned __int64 v30; // [rsp+20h] [rbp-40h] BYREF
  unsigned int v31; // [rsp+28h] [rbp-38h]

  v6 = *(unsigned int *)(a1 + 8);
  if ( (unsigned int)v6 <= 0x40 )
  {
    v17 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v6;
    result = v17 & (*(_QWORD *)a1 / *(_QWORD *)a2);
    v18 = *(_QWORD *)a1 % *(_QWORD *)a2;
    if ( *((_DWORD *)a3 + 2) > 0x40u && *a3 )
    {
      v29 = v17 & (*(_QWORD *)a1 / *(_QWORD *)a2);
      j_j___libc_free_0_0(*a3);
      result = v29;
    }
    *a3 = result;
    v19 = v18 & v17;
    *((_DWORD *)a3 + 2) = v6;
    if ( *((_DWORD *)a4 + 2) > 0x40u && *a4 )
      result = j_j___libc_free_0_0(*a4);
    *a4 = v19;
    *((_DWORD *)a4 + 2) = v6;
    return result;
  }
  v7 = ((unsigned __int64)((unsigned int)v6 - (unsigned int)sub_16A57B0(a1)) + 63) >> 6;
  if ( *(_DWORD *)(a2 + 8) > 0x40u )
  {
    v27 = *(_DWORD *)(a2 + 8);
    v8 = v27 - sub_16A57B0(a2);
    v26 = ((unsigned __int64)v8 + 63) >> 6;
    v28 = v26;
    if ( v7 )
      goto LABEL_4;
LABEL_14:
    v31 = v6;
    sub_16A4EF0((__int64)&v30, 0, 0);
    if ( *((_DWORD *)a3 + 2) > 0x40u )
    {
LABEL_34:
      if ( *a3 )
        j_j___libc_free_0_0(*a3);
    }
LABEL_36:
    *a3 = v30;
    v20 = v31;
    v31 = v6;
    *((_DWORD *)a3 + 2) = v20;
    sub_16A4EF0((__int64)&v30, 0, 0);
    if ( *((_DWORD *)a4 + 2) > 0x40u && *a4 )
      j_j___libc_free_0_0(*a4);
    *a4 = v30;
    result = v31;
    *((_DWORD *)a4 + 2) = v31;
    return result;
  }
  v26 = *(_QWORD *)a2;
  if ( !*(_QWORD *)a2 )
  {
    if ( v7 )
    {
      v28 = 0;
      goto LABEL_6;
    }
    goto LABEL_14;
  }
  _BitScanReverse64(&v13, v26);
  v26 = 1;
  v28 = 1;
  v8 = 64 - (v13 ^ 0x3F);
  if ( !v7 )
    goto LABEL_14;
LABEL_4:
  if ( v8 == 1 )
  {
    sub_16A51C0((__int64)a3, a1);
    v31 = v6;
    sub_16A4EF0((__int64)&v30, 0, 0);
    if ( *((_DWORD *)a4 + 2) > 0x40u && *a4 )
      j_j___libc_free_0_0(*a4);
    *a4 = v30;
    *((_DWORD *)a4 + 2) = v31;
  }
  if ( (unsigned int)v7 < v28 )
    goto LABEL_16;
LABEL_6:
  if ( (int)sub_16A9900(a1, (unsigned __int64 *)a2) < 0 )
  {
LABEL_16:
    if ( *((_DWORD *)a4 + 2) > 0x40u || *(_DWORD *)(a1 + 8) > 0x40u )
    {
      sub_16A51C0((__int64)a4, a1);
    }
    else
    {
      v14 = *(_QWORD *)a1;
      *a4 = *(_QWORD *)a1;
      v15 = *(unsigned int *)(a1 + 8);
      *((_DWORD *)a4 + 2) = v15;
      v16 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v15;
      if ( (unsigned int)v15 > 0x40 )
      {
        v25 = (unsigned int)((unsigned __int64)(v15 + 63) >> 6) - 1;
        *(_QWORD *)(v14 + 8 * v25) &= v16;
      }
      else
      {
        *a4 = v16 & v14;
      }
    }
    v31 = v6;
    sub_16A4EF0((__int64)&v30, 0, 0);
    if ( *((_DWORD *)a3 + 2) > 0x40u && *a3 )
      j_j___libc_free_0_0(*a3);
    *a3 = v30;
    result = v31;
    *((_DWORD *)a3 + 2) = v31;
    return result;
  }
  if ( *(_DWORD *)(a1 + 8) <= 0x40u )
  {
    if ( *(_QWORD *)a1 != *(_QWORD *)a2 )
      goto LABEL_9;
LABEL_33:
    v31 = v6;
    sub_16A4EF0((__int64)&v30, 1, 0);
    if ( *((_DWORD *)a3 + 2) > 0x40u )
      goto LABEL_34;
    goto LABEL_36;
  }
  if ( sub_16A5220(a1, (const void **)a2) )
    goto LABEL_33;
LABEL_9:
  sub_16A5130(a3, v6);
  sub_16A5130(a4, v6);
  if ( v7 != 1 )
  {
    v9 = (unsigned __int64)(v6 + 63) >> 6;
    sub_16A6110(*(__int64 **)a1, v7, *(__int64 **)a2, v28, *a3, *a4);
    memset((void *)(*a3 + 8 * v7), 0, (unsigned int)(8 * (v9 - v7)));
    v10 = 8 * ((unsigned int)v9 - v28);
    v11 = (void *)(*a4 + 8 * v26);
    return (unsigned __int64)memset(v11, 0, v10);
  }
  v21 = **(_QWORD **)a1 / **(_QWORD **)a2;
  v22 = *((_DWORD *)a3 + 2);
  v23 = **(_QWORD **)a1 % **(_QWORD **)a2;
  if ( v22 <= 0x40 )
  {
    *a3 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v22) & v21;
  }
  else
  {
    *(_QWORD *)*a3 = v21;
    memset((void *)(*a3 + 8), 0, 8 * (unsigned int)(((unsigned __int64)*((unsigned int *)a3 + 2) + 63) >> 6) - 8);
  }
  v24 = *((_DWORD *)a4 + 2);
  if ( v24 > 0x40 )
  {
    *(_QWORD *)*a4 = v23;
    v11 = (void *)(*a4 + 8);
    v10 = 8 * (unsigned int)(((unsigned __int64)*((unsigned int *)a4 + 2) + 63) >> 6) - 8;
    return (unsigned __int64)memset(v11, 0, v10);
  }
  result = v23 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v24);
  *a4 = result;
  return result;
}
