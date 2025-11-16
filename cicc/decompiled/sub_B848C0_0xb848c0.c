// Function: sub_B848C0
// Address: 0xb848c0
//
_QWORD *__fastcall sub_B848C0(_QWORD *a1)
{
  _QWORD *result; // rax
  _QWORD *v2; // rbx
  char *v3; // r13
  __int64 v4; // rax
  __int64 v5; // rsi

  *a1 = &unk_49DAC70;
  result = (_QWORD *)sub_22077B0(1304);
  v2 = result;
  if ( result )
  {
    result[1] = 0;
    v3 = (char *)(result + 71);
    *((_DWORD *)result + 22) = 1065353216;
    *result = &unk_49DAD88;
    *((_DWORD *)result + 36) = 1065353216;
    result[2] = &unk_4F81900;
    result[7] = result + 13;
    result[14] = result + 20;
    *((_DWORD *)result + 6) = 5;
    result[4] = 0;
    result[22] = &unk_49DA9F0;
    result[24] = result + 26;
    result[52] = result + 54;
    result[5] = 0;
    result[6] = 0;
    result[8] = 1;
    result[9] = 0;
    result[10] = 0;
    result[12] = 0;
    result[13] = 0;
    result[15] = 1;
    result[16] = 0;
    result[17] = 0;
    result[19] = 0;
    result[20] = 0;
    *((_BYTE *)result + 168) = 0;
    result[23] = 0;
    result[25] = 0x1000000000LL;
    result[53] = 0x1000000000LL;
    *((_DWORD *)result + 140) = 0;
    result[48] = 1;
    result[49] = 0;
    result[50] = 0;
    *((_DWORD *)result + 102) = 0;
    *((_OWORD *)result + 21) = 0;
    *((_OWORD *)result + 22) = 0;
    *((_OWORD *)result + 23) = 0;
    v4 = sub_22077B0(632);
    v5 = v4;
    if ( v4 )
    {
      *(_QWORD *)(v4 + 8) = 0;
      *(_QWORD *)(v4 + 16) = &unk_4F81901;
      *(_QWORD *)(v4 + 56) = v4 + 104;
      *(_QWORD *)(v4 + 112) = v4 + 160;
      *(_QWORD *)(v4 + 192) = v4 + 208;
      *(_QWORD *)(v4 + 416) = v4 + 432;
      *(_DWORD *)(v4 + 24) = 5;
      *(_QWORD *)(v4 + 32) = 0;
      *(_QWORD *)(v4 + 40) = 0;
      *(_QWORD *)(v4 + 48) = 0;
      *(_QWORD *)(v4 + 64) = 1;
      *(_QWORD *)(v4 + 72) = 0;
      *(_QWORD *)(v4 + 80) = 0;
      *(_QWORD *)(v4 + 96) = 0;
      *(_QWORD *)(v4 + 104) = 0;
      *(_QWORD *)(v4 + 120) = 1;
      *(_QWORD *)(v4 + 128) = 0;
      *(_QWORD *)(v4 + 136) = 0;
      *(_QWORD *)(v4 + 152) = 0;
      *(_QWORD *)(v4 + 160) = 0;
      *(_BYTE *)(v4 + 168) = 0;
      *(_QWORD *)(v4 + 184) = 0;
      *(_QWORD *)(v4 + 200) = 0x1000000000LL;
      *(_QWORD *)(v4 + 424) = 0x1000000000LL;
      *(_DWORD *)(v4 + 560) = 0;
      *(_QWORD *)(v4 + 384) = 1;
      *(_QWORD *)(v4 + 392) = 0;
      *(_QWORD *)(v4 + 400) = 0;
      *(_DWORD *)(v4 + 408) = 0;
      *(_DWORD *)(v4 + 88) = 1065353216;
      *(_DWORD *)(v4 + 144) = 1065353216;
      *(_OWORD *)(v4 + 336) = 0;
      *(_QWORD *)v4 = off_49DAB70;
      *(_QWORD *)(v4 + 176) = &unk_49DAC30;
      v5 = v4 + 176;
      *(_QWORD *)(v4 + 568) = 0;
      *(_QWORD *)(v4 + 576) = 0;
      *(_QWORD *)(v4 + 584) = 0;
      *(_DWORD *)(v4 + 592) = 0;
      *(_QWORD *)(v4 + 600) = v4 + 616;
      *(_QWORD *)(v4 + 608) = 0;
      *(_QWORD *)(v4 + 616) = 0;
      *(_QWORD *)(v4 + 624) = 0;
      *(_OWORD *)(v4 + 352) = 0;
      *(_OWORD *)(v4 + 368) = 0;
    }
    sub_B842C0((__int64)(v2 + 71), v5);
    a1[1] = v2;
    v2[161] = 0;
    v2[162] = 0;
    *v2 = &unk_49DA898;
    result = &unk_49DA990;
    v2[22] = &unk_49DA950;
    v2[71] = &unk_49DA990;
  }
  else
  {
    a1[1] = 0;
    v3 = 0;
  }
  v2[23] = v3;
  return result;
}
