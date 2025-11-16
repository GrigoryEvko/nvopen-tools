// Function: sub_C4BFE0
// Address: 0xc4bfe0
//
unsigned __int64 *__fastcall sub_C4BFE0(__int64 a1, __int64 a2, _QWORD *a3, _QWORD *a4)
{
  __int64 v6; // rbx
  unsigned __int64 v7; // r14
  unsigned int v8; // edx
  unsigned __int64 v9; // rbx
  size_t v10; // rdx
  void *v11; // rdi
  unsigned __int64 *result; // rax
  unsigned __int64 v13; // rax
  __int64 v14; // r14
  unsigned int v15; // eax
  __int64 v16; // rax
  __int64 v17; // rbx
  unsigned __int64 v18; // [rsp+8h] [rbp-58h]
  int v19; // [rsp+14h] [rbp-4Ch]
  unsigned int v20; // [rsp+14h] [rbp-4Ch]
  unsigned __int64 *v21; // [rsp+18h] [rbp-48h]
  __int64 v22; // [rsp+20h] [rbp-40h] BYREF
  unsigned int v23; // [rsp+28h] [rbp-38h]

  v6 = *(unsigned int *)(a1 + 8);
  if ( (unsigned int)v6 <= 0x40 )
  {
    result = (unsigned __int64 *)(*(_QWORD *)a1 / *(_QWORD *)a2);
    v14 = *(_QWORD *)a1 % *(_QWORD *)a2;
    if ( *((_DWORD *)a3 + 2) > 0x40u && *a3 )
    {
      v21 = (unsigned __int64 *)(*(_QWORD *)a1 / *(_QWORD *)a2);
      j_j___libc_free_0_0(*a3);
      result = v21;
    }
    *a3 = result;
    *((_DWORD *)a3 + 2) = v6;
    if ( *((_DWORD *)a4 + 2) > 0x40u && *a4 )
      result = (unsigned __int64 *)j_j___libc_free_0_0(*a4);
    *a4 = v14;
    *((_DWORD *)a4 + 2) = v6;
    return result;
  }
  v7 = ((unsigned __int64)((unsigned int)v6 - (unsigned int)sub_C444A0(a1)) + 63) >> 6;
  if ( *(_DWORD *)(a2 + 8) > 0x40u )
  {
    v19 = *(_DWORD *)(a2 + 8);
    v8 = v19 - sub_C444A0(a2);
    v18 = ((unsigned __int64)v8 + 63) >> 6;
    v20 = v18;
    if ( v7 )
      goto LABEL_4;
LABEL_14:
    v23 = v6;
    sub_C43690((__int64)&v22, 0, 0);
    if ( *((_DWORD *)a3 + 2) > 0x40u )
    {
LABEL_33:
      if ( *a3 )
        j_j___libc_free_0_0(*a3);
    }
LABEL_35:
    *a3 = v22;
    v15 = v23;
    v23 = v6;
    *((_DWORD *)a3 + 2) = v15;
    sub_C43690((__int64)&v22, 0, 0);
    if ( *((_DWORD *)a4 + 2) > 0x40u && *a4 )
      j_j___libc_free_0_0(*a4);
    *a4 = v22;
    result = (unsigned __int64 *)v23;
    *((_DWORD *)a4 + 2) = v23;
    return result;
  }
  v18 = *(_QWORD *)a2;
  if ( !*(_QWORD *)a2 )
  {
    if ( v7 )
    {
      v20 = 0;
      goto LABEL_6;
    }
    goto LABEL_14;
  }
  _BitScanReverse64(&v13, v18);
  v18 = 1;
  v20 = 1;
  v8 = 64 - (v13 ^ 0x3F);
  if ( !v7 )
    goto LABEL_14;
LABEL_4:
  if ( v8 == 1 )
  {
    sub_C43990((__int64)a3, a1);
    v23 = v6;
    sub_C43690((__int64)&v22, 0, 0);
    if ( *((_DWORD *)a4 + 2) > 0x40u && *a4 )
      j_j___libc_free_0_0(*a4);
    *a4 = v22;
    *((_DWORD *)a4 + 2) = v23;
  }
  if ( (unsigned int)v7 < v20 )
    goto LABEL_16;
LABEL_6:
  if ( (int)sub_C49970(a1, (unsigned __int64 *)a2) < 0 )
  {
LABEL_16:
    if ( *((_DWORD *)a4 + 2) > 0x40u || *(_DWORD *)(a1 + 8) > 0x40u )
    {
      sub_C43990((__int64)a4, a1);
    }
    else
    {
      *a4 = *(_QWORD *)a1;
      *((_DWORD *)a4 + 2) = *(_DWORD *)(a1 + 8);
    }
    v23 = v6;
    sub_C43690((__int64)&v22, 0, 0);
    if ( *((_DWORD *)a3 + 2) > 0x40u && *a3 )
      j_j___libc_free_0_0(*a3);
    *a3 = v22;
    result = (unsigned __int64 *)v23;
    *((_DWORD *)a3 + 2) = v23;
    return result;
  }
  if ( *(_DWORD *)(a1 + 8) <= 0x40u )
  {
    if ( *(_QWORD *)a1 != *(_QWORD *)a2 )
      goto LABEL_9;
LABEL_32:
    v23 = v6;
    sub_C43690((__int64)&v22, 1, 0);
    if ( *((_DWORD *)a3 + 2) > 0x40u )
      goto LABEL_33;
    goto LABEL_35;
  }
  if ( sub_C43C50(a1, (const void **)a2) )
    goto LABEL_32;
LABEL_9:
  sub_C43900(a3, v6);
  sub_C43900(a4, v6);
  if ( v7 != 1 )
  {
    v9 = (unsigned __int64)(v6 + 63) >> 6;
    sub_C44DF0(*(__int64 **)a1, v7, *(__int64 **)a2, v20, *a3, *a4);
    memset((void *)(*a3 + 8 * v7), 0, (unsigned int)(8 * (v9 - v7)));
    v10 = 8 * ((unsigned int)v9 - v20);
    v11 = (void *)(*a4 + 8 * v18);
    return (unsigned __int64 *)memset(v11, 0, v10);
  }
  v16 = **(_QWORD **)a1 / **(_QWORD **)a2;
  v17 = **(_QWORD **)a1 % **(_QWORD **)a2;
  if ( *((_DWORD *)a3 + 2) <= 0x40u )
  {
    *a3 = v16;
    sub_C43640(a3);
  }
  else
  {
    *(_QWORD *)*a3 = v16;
    memset((void *)(*a3 + 8LL), 0, 8 * (unsigned int)(((unsigned __int64)*((unsigned int *)a3 + 2) + 63) >> 6) - 8);
  }
  if ( *((_DWORD *)a4 + 2) > 0x40u )
  {
    *(_QWORD *)*a4 = v17;
    v11 = (void *)(*a4 + 8LL);
    v10 = 8 * (unsigned int)(((unsigned __int64)*((unsigned int *)a4 + 2) + 63) >> 6) - 8;
    return (unsigned __int64 *)memset(v11, 0, v10);
  }
  *a4 = v17;
  return sub_C43640(a4);
}
