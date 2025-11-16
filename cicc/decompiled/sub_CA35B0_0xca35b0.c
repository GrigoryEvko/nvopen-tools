// Function: sub_CA35B0
// Address: 0xca35b0
//
unsigned __int64 __fastcall sub_CA35B0(__int64 a1, __int64 a2, __int64 a3, int a4)
{
  __int64 v4; // rdx
  int i; // ebx
  __m128i si128; // xmm0
  __int64 v8; // rdx
  unsigned __int64 result; // rax

  v4 = *(_QWORD *)(a2 + 32);
  if ( a4 )
  {
    for ( i = 0; i != a4; ++i )
    {
      while ( (unsigned __int64)(*(_QWORD *)(a2 + 24) - v4) > 1 )
      {
        ++i;
        *(_WORD *)v4 = 8224;
        v4 = *(_QWORD *)(a2 + 32) + 2LL;
        *(_QWORD *)(a2 + 32) = v4;
        if ( a4 == i )
          goto LABEL_6;
      }
      sub_CB6200(a2, "  ", 2);
      v4 = *(_QWORD *)(a2 + 32);
    }
  }
LABEL_6:
  if ( (unsigned __int64)(*(_QWORD *)(a2 + 24) - v4) <= 0x14 )
  {
    sub_CB6200(a2, "RealFileSystem using ", 21);
  }
  else
  {
    si128 = _mm_load_si128((const __m128i *)&xmmword_3F6A0D0);
    *(_DWORD *)(v4 + 16) = 1735289203;
    *(_BYTE *)(v4 + 20) = 32;
    *(__m128i *)v4 = si128;
    *(_QWORD *)(a2 + 32) += 21LL;
  }
  if ( *(_BYTE *)(a1 + 328) )
    sub_904010(a2, "own");
  else
    sub_904010(a2, "process");
  v8 = *(_QWORD *)(a2 + 32);
  result = *(_QWORD *)(a2 + 24) - v8;
  if ( result <= 4 )
    return sub_CB6200(a2, " CWD\n", 5);
  *(_DWORD *)v8 = 1146569504;
  *(_BYTE *)(v8 + 4) = 10;
  *(_QWORD *)(a2 + 32) += 5LL;
  return result;
}
