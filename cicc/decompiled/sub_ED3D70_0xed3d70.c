// Function: sub_ED3D70
// Address: 0xed3d70
//
__int64 *__fastcall sub_ED3D70(__int64 *a1, __int64 a2, void *a3, size_t a4)
{
  __int64 v4; // rax
  __int64 v5; // rbx
  int v10; // eax
  unsigned int v11; // r8d
  _QWORD *v12; // r9
  __int64 **v13; // rax
  __int64 *v14; // rbx
  __int64 **v15; // rax
  __m128i *v16; // rsi
  bool v17; // zf
  _QWORD *v18; // [rsp+8h] [rbp-F8h]
  unsigned int v19; // [rsp+10h] [rbp-F0h]
  __int64 v20; // [rsp+10h] [rbp-F0h]
  _QWORD *v21; // [rsp+18h] [rbp-E8h]
  __int64 v22; // [rsp+20h] [rbp-E0h] BYREF
  __m128i v23; // [rsp+30h] [rbp-D0h] BYREF
  __int64 v24; // [rsp+40h] [rbp-C0h]
  char v25; // [rsp+50h] [rbp-B0h]
  char v26; // [rsp+51h] [rbp-AFh]

  if ( a4 )
  {
    v10 = sub_C92610();
    v11 = sub_C92740(a2 + 24, a3, a4, v10);
    v12 = (_QWORD *)(*(_QWORD *)(a2 + 24) + 8LL * v11);
    if ( *v12 )
    {
      if ( *v12 != -8 )
      {
        *a1 = 1;
        return a1;
      }
      --*(_DWORD *)(a2 + 40);
    }
    v18 = v12;
    v19 = v11;
    v21 = (_QWORD *)sub_C7D670(a4 + 9, 8);
    memcpy(v21 + 1, a3, a4);
    *((_BYTE *)v21 + a4 + 8) = 0;
    *v21 = a4;
    *v18 = v21;
    ++*(_DWORD *)(a2 + 36);
    v13 = (__int64 **)(*(_QWORD *)(a2 + 24) + 8LL * (unsigned int)sub_C929D0((__int64 *)(a2 + 24), v19));
    v14 = *v13;
    if ( !*v13 || v14 == (__int64 *)-8LL )
    {
      v15 = v13 + 1;
      do
      {
        do
          v14 = *v15++;
        while ( !v14 );
      }
      while ( v14 == (__int64 *)-8LL );
    }
    v20 = *v14;
    sub_C7D030(&v23);
    sub_C7D280(v23.m128i_i32, (int *)a3, a4);
    sub_C7D290(&v23, &v22);
    v16 = *(__m128i **)(a2 + 80);
    v23.m128i_i64[1] = (__int64)(v14 + 1);
    v17 = v16 == *(__m128i **)(a2 + 88);
    v23.m128i_i64[0] = v22;
    v24 = v20;
    if ( v17 )
    {
      sub_ED3BC0((const __m128i **)(a2 + 72), v16, &v23);
    }
    else
    {
      if ( v16 )
      {
        *v16 = _mm_loadu_si128(&v23);
        v16[1].m128i_i64[0] = v24;
        v16 = *(__m128i **)(a2 + 80);
      }
      *(_QWORD *)(a2 + 80) = (char *)v16 + 24;
    }
    *(_BYTE *)(a2 + 392) = 0;
    *a1 = 1;
  }
  else
  {
    v26 = 1;
    v23.m128i_i64[0] = (__int64)"symbol name is empty";
    v25 = 3;
    v4 = sub_22077B0(48);
    v5 = v4;
    if ( v4 )
    {
      *(_DWORD *)(v4 + 8) = 9;
      *(_QWORD *)v4 = &unk_49E4BC8;
      sub_CA0F50((__int64 *)(v4 + 16), (void **)&v23);
    }
    *a1 = v5 | 1;
  }
  return a1;
}
