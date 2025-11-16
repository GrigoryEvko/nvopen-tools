// Function: sub_87E420
// Address: 0x87e420
//
__int64 __fastcall sub_87E420(char a1)
{
  __int64 v1; // rax
  __int64 v2; // r12

  v1 = sub_823970(432);
  *(_DWORD *)(v1 + 40) = 0;
  v2 = v1;
  *(_QWORD *)(v1 + 48) = 0;
  *(_QWORD *)(v1 + 56) = 0;
  *(_DWORD *)(v1 + 64) = 0;
  sub_879020(v1, 1);
  *(_QWORD *)(v2 + 72) = 0;
  *(_QWORD *)(v2 + 80) = 0;
  *(_QWORD *)(v2 + 88) = 0;
  *(_QWORD *)(v2 + 96) = 0;
  *(_QWORD *)(v2 + 104) = 0;
  *(_QWORD *)(v2 + 112) = 0;
  *(_QWORD *)(v2 + 120) = 0;
  *(_QWORD *)(v2 + 128) = 0;
  *(_QWORD *)(v2 + 136) = 0;
  *(_QWORD *)(v2 + 144) = 0;
  *(_QWORD *)(v2 + 152) = 0;
  *(_BYTE *)(v2 + 160) = 0;
  switch ( a1 )
  {
    case 4:
    case 5:
    case 6:
    case 19:
      *(_DWORD *)(v2 + 264) &= 0xF8000000;
      *(_QWORD *)(v2 + 168) = 0;
      *(_QWORD *)(v2 + 176) = 0;
      *(_QWORD *)(v2 + 184) = 0;
      *(_QWORD *)(v2 + 192) = 0;
      *(_QWORD *)(v2 + 200) = 0;
      *(_QWORD *)(v2 + 208) = 0;
      *(_QWORD *)(v2 + 216) = 0;
      sub_879020(v2 + 224, 1);
      return v2;
    case 9:
    case 21:
      *(_BYTE *)(v2 + 168) &= ~1u;
      *(_QWORD *)(v2 + 176) = 0;
      *(_QWORD *)(v2 + 184) = 0;
      *(_QWORD *)(v2 + 192) = 0;
      sub_879020(v2 + 200, 1);
      *(_DWORD *)(v2 + 240) = 0;
      return v2;
    case 10:
    case 20:
      *(_QWORD *)(v2 + 168) = 0;
      *(_QWORD *)(v2 + 176) = 0;
      sub_87E3B0(v2 + 184);
      *(_QWORD *)(v2 + 288) = 0;
      sub_879020(v2 + 296, 1);
      sub_879020(v2 + 336, 1);
      *(_BYTE *)(v2 + 424) &= 0xE0u;
      *(_QWORD *)(v2 + 376) = 0;
      *(_WORD *)(v2 + 384) = 0;
      *(_QWORD *)(v2 + 388) = 0;
      *(_QWORD *)(v2 + 400) = 0;
      *(_QWORD *)(v2 + 408) = 0;
      *(_QWORD *)(v2 + 416) = 0;
      return v2;
    case 22:
      return v2;
    default:
      sub_721090();
  }
}
