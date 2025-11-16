// Function: sub_1F4AD00
// Address: 0x1f4ad00
//
__int64 __fastcall sub_1F4AD00(__int64 a1, __int64 a2, __int64 a3, _QWORD *a4)
{
  int v7; // ebx
  unsigned int v8; // ebx
  __int64 v9; // rax
  size_t v10; // r9
  size_t v11; // rdx
  _QWORD *v12; // rcx
  void *v13; // r15
  __int64 *v14; // rsi
  int v15; // eax
  unsigned int v16; // ebx
  __int64 v17; // rdx
  unsigned __int64 v18; // rcx
  unsigned int v19; // eax
  unsigned int v20; // edi
  __int64 v21; // rdx
  _QWORD *v22; // rcx
  __int64 v23; // rsi
  __int64 v24; // rcx
  __int64 **v26; // r15
  __int64 **i; // rbx
  __int64 v28; // rax
  _QWORD *v30; // [rsp+8h] [rbp-68h]
  size_t v31; // [rsp+10h] [rbp-60h]
  size_t n; // [rsp+18h] [rbp-58h]
  size_t na; // [rsp+18h] [rbp-58h]
  unsigned __int64 v34; // [rsp+20h] [rbp-50h] BYREF
  unsigned __int64 v35; // [rsp+28h] [rbp-48h]
  int v36; // [rsp+30h] [rbp-40h]

  v7 = *(_DWORD *)(a2 + 16);
  *(_QWORD *)a1 = 0;
  *(_QWORD *)(a1 + 8) = 0;
  *(_DWORD *)(a1 + 16) = v7;
  v8 = (unsigned int)(v7 + 63) >> 6;
  v9 = malloc(8LL * v8);
  v10 = 8LL * v8;
  v11 = v8;
  v12 = a4;
  v13 = (void *)v9;
  if ( !v9 )
  {
    if ( 8LL * v8 || (v28 = malloc(1u), v11 = v8, v10 = 0, v12 = a4, !v28) )
    {
      v30 = v12;
      v31 = v10;
      na = v11;
      sub_16BD1C0("Allocation failed", 1u);
      v11 = na;
      v10 = v31;
      v12 = v30;
    }
    else
    {
      v13 = (void *)v28;
    }
  }
  *(_QWORD *)a1 = v13;
  *(_QWORD *)(a1 + 8) = v11;
  if ( v8 )
  {
    n = (size_t)v12;
    memset(v13, 0, v10);
    v12 = (_QWORD *)n;
  }
  if ( v12 )
  {
    v14 = sub_1F4AAF0(a2, v12);
    if ( v14 )
      sub_1F49DC0(a3, v14, (_QWORD *)a1);
  }
  else
  {
    v26 = *(__int64 ***)(a2 + 264);
    for ( i = *(__int64 ***)(a2 + 256); v26 != i; ++i )
    {
      if ( *(_BYTE *)(**i + 29) )
        sub_1F49DC0(a3, *i, (_QWORD *)a1);
    }
  }
  (*(void (__fastcall **)(unsigned __int64 *, __int64, __int64))(*(_QWORD *)a2 + 64LL))(&v34, a2, a3);
  v15 = v36;
  v16 = (unsigned int)(v36 + 63) >> 6;
  if ( v16 )
  {
    v17 = 0;
    do
    {
      *(_QWORD *)(v34 + 8 * v17) = ~*(_QWORD *)(v34 + 8 * v17);
      v15 = v36;
      ++v17;
      v16 = (unsigned int)(v36 + 63) >> 6;
    }
    while ( v16 > (unsigned int)v17 );
    v18 = v16;
  }
  else
  {
    v18 = 0;
  }
  if ( v35 > v18 )
  {
    memset((void *)(v34 + 8 * v18), 0, 8 * (v35 - v18));
    v15 = v36;
  }
  if ( (v15 & 0x3F) != 0 )
  {
    *(_QWORD *)(v34 + 8LL * (v16 - 1)) &= ~(-1LL << (v15 & 0x3F));
    v15 = v36;
  }
  v19 = (unsigned int)(v15 + 63) >> 6;
  v20 = (unsigned int)(*(_DWORD *)(a1 + 16) + 63) >> 6;
  if ( v19 > v20 )
    v19 = v20;
  v21 = 0;
  if ( v19 )
  {
    do
    {
      v22 = (_QWORD *)(v21 + *(_QWORD *)a1);
      v23 = *(_QWORD *)(v34 + v21);
      v21 += 8;
      *v22 &= v23;
    }
    while ( 8LL * v19 != v21 );
  }
  for ( ; v20 != v19; *(_QWORD *)(*(_QWORD *)a1 + 8 * v24) = 0 )
    v24 = v19++;
  _libc_free(v34);
  return a1;
}
