// Function: sub_37EA930
// Address: 0x37ea930
//
char __fastcall sub_37EA930(__int64 a1, __int64 a2)
{
  __int64 v4; // r12
  unsigned int v5; // ebx
  int i; // r13d
  __int64 v7; // rax
  __int64 v8; // rdi
  __int64 (*v9)(); // rax
  __int64 v10; // rax
  __int64 v11; // r12
  __int64 j; // rbx
  __int64 v13; // rdi
  __int64 v14; // rdi
  unsigned int v15; // eax
  __m128i *v16; // rsi
  unsigned int v18; // [rsp+Ch] [rbp-44h]
  __m128i v19; // [rsp+10h] [rbp-40h] BYREF

  v4 = *(_QWORD *)(a2 + 16);
  v5 = *(unsigned __int8 *)(v4 + 4);
  for ( i = *(unsigned __int16 *)(v4 + 2); v5 != i; ++v5 )
  {
    v7 = *(_QWORD *)(a2 + 32) + 40LL * v5;
    if ( !*(_BYTE *)v7 )
    {
      if ( *(_DWORD *)(v7 + 8) )
      {
        if ( (*(_BYTE *)(v7 + 3) & 0x10) == 0 && (*(_BYTE *)(v7 + 4) & 1) != 0 )
        {
          v8 = *(_QWORD *)(a1 + 208);
          v9 = *(__int64 (**)())(*(_QWORD *)v8 + 1240LL);
          if ( v9 != sub_2FDC790 )
          {
            v15 = ((__int64 (__fastcall *)(__int64, __int64, _QWORD, _QWORD))v9)(v8, a2, v5, *(_QWORD *)(a1 + 216));
            if ( v15 )
            {
              v18 = v15;
              if ( !(unsigned __int8)sub_37EA240(a1, a2, v5, v15) && sub_37EA510(a1, a2, v5, v18) )
              {
                v19.m128i_i64[0] = a2;
                v16 = *(__m128i **)(a1 + 552);
                v19.m128i_i32[2] = v5;
                if ( v16 == *(__m128i **)(a1 + 560) )
                {
                  sub_37EA7B0((unsigned __int64 *)(a1 + 544), v16, &v19);
                }
                else
                {
                  if ( v16 )
                  {
                    *v16 = _mm_loadu_si128(&v19);
                    v16 = *(__m128i **)(a1 + 552);
                  }
                  *(_QWORD *)(a1 + 552) = v16 + 1;
                }
              }
            }
          }
        }
      }
    }
  }
  LOBYTE(v10) = sub_B2D610(**(_QWORD **)(a1 + 200), 18);
  if ( !(_BYTE)v10 )
  {
    v10 = *(_QWORD *)(a2 + 16);
    v11 = (*(_BYTE *)(v10 + 24) & 2) != 0 ? *(_DWORD *)(a2 + 40) & 0xFFFFFF : *(unsigned __int8 *)(v4 + 4);
    if ( (_DWORD)v11 )
    {
      for ( j = 0; j != v11; ++j )
      {
        v10 = *(_QWORD *)(a2 + 32) + 40 * j;
        if ( !*(_BYTE *)v10 )
        {
          if ( *(_DWORD *)(v10 + 8) )
          {
            if ( (*(_BYTE *)(v10 + 3) & 0x10) != 0 )
            {
              v13 = *(_QWORD *)(a1 + 208);
              v10 = *(_QWORD *)(*(_QWORD *)v13 + 1232LL);
              if ( (__int64 (*)())v10 != sub_2FDC780 )
              {
                LODWORD(v10) = ((__int64 (__fastcall *)(__int64, __int64, _QWORD, _QWORD))v10)(
                                 v13,
                                 a2,
                                 (unsigned int)j,
                                 *(_QWORD *)(a1 + 216));
                if ( (_DWORD)v10 )
                {
                  LOBYTE(v10) = sub_37EA510(a1, a2, j, v10);
                  if ( (_BYTE)v10 )
                  {
                    v14 = *(_QWORD *)(a1 + 208);
                    v10 = *(_QWORD *)(*(_QWORD *)v14 + 1248LL);
                    if ( (void (*)())v10 != nullsub_1686 )
                      LOBYTE(v10) = ((__int64 (__fastcall *)(__int64, __int64, _QWORD, _QWORD))v10)(
                                      v14,
                                      a2,
                                      (unsigned int)j,
                                      *(_QWORD *)(a1 + 216));
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  return v10;
}
