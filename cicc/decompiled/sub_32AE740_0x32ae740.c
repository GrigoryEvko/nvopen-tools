// Function: sub_32AE740
// Address: 0x32ae740
//
char __fastcall sub_32AE740(int *a1, __int64 a2, __int64 a3)
{
  int v5; // r14d
  __int64 v6; // rdi
  char result; // al
  unsigned int v8; // r14d
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 *v11; // rax
  int v12; // r15d
  __int64 v13; // r14
  __int64 v14; // rdi
  __int64 v15; // rax
  unsigned int v16; // r15d
  __int64 v17; // rax
  __int64 v18; // rax
  __int64 v19; // rax
  __m128i v20; // [rsp+0h] [rbp-60h]
  __m128i v21; // [rsp+10h] [rbp-50h]
  __int64 v22; // [rsp+20h] [rbp-40h]
  __int64 v23; // [rsp+20h] [rbp-40h]
  __int64 v24; // [rsp+28h] [rbp-38h]
  __int64 v25; // [rsp+28h] [rbp-38h]
  __int64 v26; // [rsp+28h] [rbp-38h]
  __int64 v27; // [rsp+28h] [rbp-38h]

  v5 = *a1;
  if ( (unsigned __int8)sub_33CB110(*(unsigned int *)(a3 + 24)) )
  {
    v22 = sub_33CB280(*(unsigned int *)(a3 + 24), ((unsigned __int8)(*(_DWORD *)(a3 + 28) >> 12) ^ 1) & 1);
    if ( !BYTE4(v22) || v5 != (_DWORD)v22 )
      return 0;
    v8 = *(_DWORD *)(a3 + 24);
    v24 = sub_33CB160(v8);
    if ( BYTE4(v24) )
    {
      v9 = *(_QWORD *)(a3 + 40) + 40LL * (unsigned int)v24;
      if ( *(_QWORD *)v9 != *(_QWORD *)(a2 + 16) || *(_DWORD *)(v9 + 8) != *(_DWORD *)(a2 + 24) )
      {
        result = sub_33D1720(*(_QWORD *)v9, 0);
        if ( !result )
          return result;
      }
    }
    v25 = sub_33CB1F0(v8);
    if ( BYTE4(v25) )
    {
      v10 = *(_QWORD *)(a3 + 40) + 40LL * (unsigned int)v25;
      if ( *(_QWORD *)(a2 + 32) != *(_QWORD *)v10 || *(_DWORD *)(a2 + 40) != *(_DWORD *)(v10 + 8) )
        return 0;
    }
    v6 = *(unsigned int *)(a3 + 24);
  }
  else
  {
    v6 = *(unsigned int *)(a3 + 24);
    if ( v5 != (_DWORD)v6 )
      return 0;
  }
  sub_33CB110(v6);
  v11 = *(__int64 **)(a3 + 40);
  v12 = a1[2];
  v13 = *v11;
  if ( (unsigned __int8)sub_33CB110(*(unsigned int *)(*v11 + 24)) )
  {
    v23 = sub_33CB280(*(unsigned int *)(v13 + 24), ((unsigned __int8)(*(_DWORD *)(v13 + 28) >> 12) ^ 1) & 1);
    if ( !BYTE4(v23) || v12 != (_DWORD)v23 )
      return 0;
    v16 = *(_DWORD *)(v13 + 24);
    v26 = sub_33CB160(v16);
    if ( BYTE4(v26) )
    {
      v17 = *(_QWORD *)(v13 + 40) + 40LL * (unsigned int)v26;
      if ( (*(_QWORD *)v17 != *(_QWORD *)(a2 + 16) || *(_DWORD *)(v17 + 8) != *(_DWORD *)(a2 + 24))
        && !(unsigned __int8)sub_33D1720(*(_QWORD *)v17, 0) )
      {
        return 0;
      }
    }
    v27 = sub_33CB1F0(v16);
    if ( BYTE4(v27) )
    {
      v18 = *(_QWORD *)(v13 + 40) + 40LL * (unsigned int)v27;
      if ( *(_QWORD *)(a2 + 32) != *(_QWORD *)v18 || *(_DWORD *)(a2 + 40) != *(_DWORD *)(v18 + 8) )
        return 0;
    }
    v14 = *(unsigned int *)(v13 + 24);
  }
  else
  {
    v14 = *(unsigned int *)(v13 + 24);
    if ( v12 != (_DWORD)v14 )
      return 0;
  }
  sub_33CB110(v14);
  v15 = *((_QWORD *)a1 + 2);
  v21 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(v13 + 40));
  *(_QWORD *)v15 = v21.m128i_i64[0];
  *(_DWORD *)(v15 + 8) = v21.m128i_i32[2];
  if ( !(unsigned __int8)sub_33E07E0(
                           *(_QWORD *)(*(_QWORD *)(v13 + 40) + 40LL),
                           *(_QWORD *)(*(_QWORD *)(v13 + 40) + 48LL),
                           0) )
  {
    v19 = *((_QWORD *)a1 + 2);
    v20 = _mm_loadu_si128((const __m128i *)(*(_QWORD *)(v13 + 40) + 40LL));
    *(_QWORD *)v19 = v20.m128i_i64[0];
    *(_DWORD *)(v19 + 8) = v20.m128i_i32[2];
    if ( !(unsigned __int8)sub_33E07E0(**(_QWORD **)(v13 + 40), *(_QWORD *)(*(_QWORD *)(v13 + 40) + 8LL), 0) )
      return 0;
  }
  if ( *((_BYTE *)a1 + 32) && a1[7] != (a1[7] & *(_DWORD *)(v13 + 28)) )
    return 0;
  result = 1;
  if ( *((_BYTE *)a1 + 44) )
    return (a1[10] & *(_DWORD *)(a3 + 28)) == a1[10];
  return result;
}
