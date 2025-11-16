// Function: sub_135D320
// Address: 0x135d320
//
void __fastcall sub_135D320(__int64 a1, __int64 a2)
{
  __int64 v2; // r14
  __int64 v3; // r15
  __int64 v4; // rax
  unsigned __int64 v5; // rdx
  __int64 v6; // r13
  __int64 v7; // rbx
  __int64 v8; // rsi
  __int64 i; // rbx
  __m128i v10; // xmm0
  __int64 v11; // rax
  __int64 v12; // rax
  char v13; // r8
  __m128i v14; // [rsp+0h] [rbp-50h] BYREF
  __int64 v15; // [rsp+10h] [rbp-40h]

  v2 = a2 + 8;
  v3 = *(_QWORD *)(a2 + 16);
  if ( v3 != a2 + 8 )
  {
    do
    {
      if ( !*(_QWORD *)(v3 + 32) )
      {
        v4 = *(_QWORD *)(v3 + 40);
        v5 = 0xAAAAAAAAAAAAAAABLL * ((*(_QWORD *)(v3 + 48) - v4) >> 3);
        if ( (_DWORD)v5 )
        {
          v6 = 0;
          v7 = 24LL * (unsigned int)v5;
          while ( 1 )
          {
            v8 = *(_QWORD *)(v4 + v6 + 16);
            if ( v8 )
              sub_135CDE0(a1, v8);
            v6 += 24;
            if ( v7 == v6 )
              break;
            v4 = *(_QWORD *)(v3 + 40);
          }
        }
        for ( i = *(_QWORD *)(v3 + 16); i; i = *(_QWORD *)(i + 16) )
        {
          v12 = *(_QWORD *)(i + 40);
          v13 = (*(_BYTE *)(v3 + 67) >> 4) & 3;
          if ( v12 != -8 && v12 != -16 || *(_QWORD *)(i + 48) || *(_QWORD *)(i + 56) )
          {
            v10 = _mm_loadu_si128((const __m128i *)(i + 40));
            v15 = *(_QWORD *)(i + 56);
            v14 = v10;
          }
          else
          {
            v14 = 0u;
            v15 = 0;
          }
          v11 = sub_135C460(a1, *(_QWORD *)i, *(_QWORD *)(i + 32), &v14, v13);
          if ( *(char *)(v3 + 67) < 0 )
            *(_BYTE *)(v11 + 67) |= 0x80u;
        }
      }
      v3 = *(_QWORD *)(v3 + 8);
    }
    while ( v2 != v3 );
  }
}
