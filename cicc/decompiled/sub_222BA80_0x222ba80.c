// Function: sub_222BA80
// Address: 0x222ba80
//
__int64 __fastcall sub_222BA80(__int64 a1)
{
  __int64 result; // rax

  *(_QWORD *)a1 = off_4A07480;
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 32) = 0;
  *(_QWORD *)(a1 + 40) = 0;
  *(_QWORD *)(a1 + 48) = 0;
  sub_220A990((volatile signed __int32 **)(a1 + 56));
  *(_QWORD *)(a1 + 96) = 0;
  *(_OWORD *)(a1 + 64) = 0;
  *(_OWORD *)(a1 + 80) = 0;
  *(_QWORD *)a1 = off_4A06448;
  sub_2207CC0(a1 + 104);
  *(_QWORD *)(a1 + 120) = 0;
  *(_QWORD *)(a1 + 128) = 0;
  *(_QWORD *)(a1 + 136) = 0;
  *(_DWORD *)(a1 + 144) = 0;
  *(_QWORD *)(a1 + 152) = 0;
  *(_QWORD *)(a1 + 160) = 0x2000;
  *(_DWORD *)(a1 + 168) = 0;
  *(_QWORD *)(a1 + 176) = 0;
  *(_QWORD *)(a1 + 184) = 0;
  *(_BYTE *)(a1 + 192) = 0;
  *(_QWORD *)(a1 + 200) = 0;
  *(_QWORD *)(a1 + 208) = 0;
  *(_QWORD *)(a1 + 216) = 0;
  *(_QWORD *)(a1 + 224) = 0;
  *(_QWORD *)(a1 + 232) = 0;
  result = sub_2231390(a1 + 56, a1 + 64);
  if ( (_BYTE)result )
  {
    result = sub_222FD10(a1 + 56);
    *(_QWORD *)(a1 + 200) = result;
  }
  return result;
}
