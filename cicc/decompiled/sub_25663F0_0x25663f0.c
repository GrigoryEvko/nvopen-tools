// Function: sub_25663F0
// Address: 0x25663f0
//
__int64 __fastcall sub_25663F0(__m128i *a1, __int64 a2)
{
  char v2; // al
  __int64 result; // rax
  __m128i v4; // xmm0

  v2 = sub_2509800(a1);
  if ( v2 != 4 )
  {
    if ( v2 > 4 )
    {
      if ( (unsigned __int8)(v2 - 5) > 2u )
        return 0;
    }
    else if ( (unsigned __int8)v2 > 3u )
    {
      return 0;
    }
    BUG();
  }
  result = sub_A777F0(0x68u, *(__int64 **)(a2 + 128));
  if ( result )
  {
    v4 = _mm_loadu_si128(a1);
    *(_QWORD *)(result + 8) = 0;
    *(_QWORD *)(result + 40) = result + 56;
    *(_WORD *)(result + 96) = 256;
    *(_QWORD *)(result + 48) = 0x200000000LL;
    *(_QWORD *)(result + 16) = 0;
    *(_QWORD *)(result + 24) = 0;
    *(_DWORD *)(result + 32) = 0;
    *(_QWORD *)result = off_4A18C48;
    *(_QWORD *)(result + 88) = &unk_4A18CC8;
    *(__m128i *)(result + 72) = v4;
  }
  return result;
}
