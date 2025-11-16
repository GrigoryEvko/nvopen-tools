// Function: sub_25623E0
// Address: 0x25623e0
//
__int64 __fastcall sub_25623E0(__m128i *a1, __int64 a2)
{
  char v2; // al
  __int64 v3; // rax
  __int64 v4; // r12
  __int64 (__fastcall **v5)(); // rax
  __int64 (__fastcall **v6)(); // rax
  __int64 v8; // rax

  v2 = sub_2509800(a1);
  if ( v2 == 5 )
  {
    v8 = sub_A777F0(0xB0u, *(__int64 **)(a2 + 128));
    v4 = v8;
    if ( !v8 )
      return v4;
    *(__m128i *)(v8 + 72) = _mm_loadu_si128(a1);
    sub_2553350(v8);
    *(_QWORD *)(v4 + 112) = a2;
    *(_WORD *)(v4 + 96) = 256;
    *(_QWORD *)(v4 + 152) = v4 + 168;
    v5 = off_4A1D250;
    *(_QWORD *)(v4 + 120) = 0;
    *(_QWORD *)(v4 + 128) = 0;
    *(_QWORD *)(v4 + 136) = 0;
    *(_DWORD *)(v4 + 144) = 0;
    *(_QWORD *)(v4 + 160) = 0;
    *(_WORD *)(v4 + 168) = 0;
    goto LABEL_6;
  }
  if ( v2 > 5 )
  {
    if ( (unsigned __int8)(v2 - 6) > 1u )
      return 0;
LABEL_14:
    BUG();
  }
  if ( v2 == 4 )
  {
    v3 = sub_A777F0(0xB0u, *(__int64 **)(a2 + 128));
    v4 = v3;
    if ( !v3 )
      return v4;
    *(__m128i *)(v3 + 72) = _mm_loadu_si128(a1);
    sub_2553350(v3);
    *(_QWORD *)(v4 + 112) = a2;
    *(_WORD *)(v4 + 96) = 256;
    *(_QWORD *)(v4 + 120) = 0;
    *(_QWORD *)(v4 + 128) = 0;
    *(_QWORD *)(v4 + 136) = 0;
    *(_DWORD *)(v4 + 144) = 0;
    *(_QWORD *)(v4 + 160) = 0;
    *(_WORD *)(v4 + 168) = 0;
    *(_QWORD *)(v4 + 152) = v4 + 168;
    v5 = off_4A1D388;
LABEL_6:
    *(_QWORD *)v4 = v5;
    v6 = v5 + 21;
    *(_QWORD *)(v4 + 88) = v6;
    *(_QWORD *)(v4 + 104) = v6 + 12;
    return v4;
  }
  if ( v2 >= 0 )
    goto LABEL_14;
  return 0;
}
