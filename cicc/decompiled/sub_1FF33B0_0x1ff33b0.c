// Function: sub_1FF33B0
// Address: 0x1ff33b0
//
__int64 __fastcall sub_1FF33B0(__int64 a1, __int64 a2, int a3, int a4, int a5, int a6, int a7)
{
  int v11; // edx
  __int64 v12; // rax

  v11 = a7;
  if ( (unsigned int)(*(__int16 *)(a2 + 24) - 81) <= 0x11 )
  {
    v12 = sub_1D44F30(*(const __m128i **)(a1 + 16), a2);
    v11 = a7;
    a2 = v12;
  }
  switch ( **(_BYTE **)(a2 + 40) )
  {
    case 9:
      return sub_1FF2B80(a1, a3, a2, 0);
    case 0xA:
      a3 = a4;
      break;
    case 0xB:
      a3 = a5;
      break;
    case 0xC:
      a3 = a6;
      break;
    case 0xD:
      a3 = v11;
      break;
  }
  return sub_1FF2B80(a1, a3, a2, 0);
}
