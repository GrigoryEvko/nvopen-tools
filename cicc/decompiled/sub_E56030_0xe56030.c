// Function: sub_E56030
// Address: 0xe56030
//
__int64 __fastcall sub_E56030(__int64 a1, __int64 a2)
{
  __int64 v3; // r13
  __int64 v4; // r12
  __int64 v5; // rdi
  __int64 v6; // rsi
  __int64 v7; // rdi
  __int64 v8; // rdi
  __int64 v9; // r8
  __int64 v10; // r13
  __int64 v11; // r13
  __int64 v12; // r12
  _QWORD *v13; // rdi
  _QWORD *v14; // r13
  _QWORD *v15; // r12
  __int64 result; // rax
  _QWORD *v17; // r13
  _QWORD *v18; // r12

  v3 = *(_QWORD *)(a1 + 552);
  v4 = v3 + 32LL * *(unsigned int *)(a1 + 560);
  if ( v3 != v4 )
  {
    do
    {
      v5 = *(_QWORD *)(v4 - 24);
      v4 -= 32;
      if ( v5 )
      {
        a2 = *(_QWORD *)(v4 + 24) - v5;
        j_j___libc_free_0(v5, a2);
      }
    }
    while ( v3 != v4 );
    v4 = *(_QWORD *)(a1 + 552);
  }
  if ( v4 != a1 + 568 )
    _libc_free(v4, a2);
  v6 = 16LL * *(unsigned int *)(a1 + 544);
  sub_C7D6A0(*(_QWORD *)(a1 + 528), v6, 8);
  v7 = *(_QWORD *)(a1 + 432);
  if ( v7 != a1 + 448 )
  {
    v6 = *(_QWORD *)(a1 + 448) + 1LL;
    j_j___libc_free_0(v7, v6);
  }
  v8 = *(_QWORD *)(a1 + 400);
  if ( v8 != a1 + 416 )
  {
    v6 = *(_QWORD *)(a1 + 416) + 1LL;
    j_j___libc_free_0(v8, v6);
  }
  v9 = *(_QWORD *)(a1 + 376);
  if ( *(_DWORD *)(a1 + 388) )
  {
    v10 = *(unsigned int *)(a1 + 384);
    if ( (_DWORD)v10 )
    {
      v11 = 8 * v10;
      v12 = 0;
      do
      {
        v13 = *(_QWORD **)(v9 + v12);
        if ( v13 != (_QWORD *)-8LL && v13 )
        {
          v6 = *v13 + 17LL;
          sub_C7D6A0((__int64)v13, v6, 8);
          v9 = *(_QWORD *)(a1 + 376);
        }
        v12 += 8;
      }
      while ( v11 != v12 );
    }
  }
  _libc_free(v9, v6);
  v14 = *(_QWORD **)(a1 + 120);
  v15 = &v14[10 * *(unsigned int *)(a1 + 128)];
  if ( v14 != v15 )
  {
    do
    {
      v15 -= 10;
      if ( (_QWORD *)*v15 != v15 + 2 )
      {
        v6 = v15[2] + 1LL;
        j_j___libc_free_0(*v15, v6);
      }
    }
    while ( v14 != v15 );
    v15 = *(_QWORD **)(a1 + 120);
  }
  result = a1 + 136;
  if ( v15 != (_QWORD *)(a1 + 136) )
    result = _libc_free(v15, v6);
  v17 = *(_QWORD **)(a1 + 8);
  v18 = &v17[4 * *(unsigned int *)(a1 + 16)];
  if ( v17 != v18 )
  {
    do
    {
      v18 -= 4;
      result = (__int64)(v18 + 2);
      if ( (_QWORD *)*v18 != v18 + 2 )
      {
        v6 = v18[2] + 1LL;
        result = j_j___libc_free_0(*v18, v6);
      }
    }
    while ( v17 != v18 );
    v18 = *(_QWORD **)(a1 + 8);
  }
  if ( v18 != (_QWORD *)(a1 + 24) )
    return _libc_free(v18, v6);
  return result;
}
