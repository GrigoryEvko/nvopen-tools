// Function: sub_1611730
// Address: 0x1611730
//
_QWORD *__fastcall sub_1611730(_QWORD *a1, __int64 a2)
{
  _QWORD *result; // rax
  _QWORD *v3; // rbx
  char *v4; // r13
  __int64 v5; // rax
  __int64 v6; // rsi

  *a1 = &unk_49EDE08;
  a1[1] = a2;
  result = (_QWORD *)sub_22077B0(1320);
  v3 = result;
  if ( result )
  {
    result[1] = 0;
    *((_DWORD *)result + 6) = 6;
    v4 = (char *)(result + 71);
    result[4] = 0;
    *result = &unk_49EDE80;
    result[5] = 0;
    result[2] = &unk_4F9E3B8;
    result[10] = result + 8;
    result[11] = result + 8;
    result[16] = result + 14;
    result[17] = result + 14;
    result[6] = 0;
    *((_DWORD *)result + 16) = 0;
    result[20] = &unk_49EDA68;
    result[23] = result + 25;
    result[52] = result + 54;
    result[9] = 0;
    result[12] = 0;
    *((_DWORD *)result + 28) = 0;
    result[15] = 0;
    result[18] = 0;
    *((_BYTE *)result + 152) = 0;
    result[21] = 0;
    result[22] = 0;
    result[24] = 0x1000000000LL;
    result[53] = 0x1000000000LL;
    *((_DWORD *)result + 140) = 0;
    result[41] = 0;
    result[42] = 0;
    result[43] = 0;
    result[44] = 0;
    result[45] = 0;
    result[46] = 0;
    result[47] = 0;
    result[48] = 1;
    result[49] = 0;
    result[50] = 0;
    *((_DWORD *)result + 102) = 0;
    v5 = sub_22077B0(640);
    v6 = v5;
    if ( v5 )
    {
      *(_QWORD *)(v5 + 8) = 0;
      *(_QWORD *)(v5 + 16) = &unk_4F9E3B9;
      *(_QWORD *)(v5 + 80) = v5 + 64;
      *(_QWORD *)(v5 + 88) = v5 + 64;
      *(_QWORD *)(v5 + 128) = v5 + 112;
      *(_QWORD *)(v5 + 136) = v5 + 112;
      *(_QWORD *)(v5 + 184) = v5 + 200;
      v6 = v5 + 160;
      *(_QWORD *)(v5 + 416) = v5 + 432;
      *(_DWORD *)(v5 + 24) = 6;
      *(_QWORD *)(v5 + 32) = 0;
      *(_QWORD *)(v5 + 40) = 0;
      *(_QWORD *)(v5 + 48) = 0;
      *(_DWORD *)(v5 + 64) = 0;
      *(_QWORD *)(v5 + 72) = 0;
      *(_QWORD *)(v5 + 96) = 0;
      *(_DWORD *)(v5 + 112) = 0;
      *(_QWORD *)(v5 + 120) = 0;
      *(_QWORD *)(v5 + 144) = 0;
      *(_BYTE *)(v5 + 152) = 0;
      *(_QWORD *)(v5 + 168) = 0;
      *(_QWORD *)(v5 + 176) = 0;
      *(_QWORD *)(v5 + 192) = 0x1000000000LL;
      *(_QWORD *)(v5 + 424) = 0x1000000000LL;
      *(_DWORD *)(v5 + 560) = 0;
      *(_QWORD *)(v5 + 328) = 0;
      *(_QWORD *)(v5 + 336) = 0;
      *(_QWORD *)(v5 + 344) = 0;
      *(_QWORD *)(v5 + 352) = 0;
      *(_QWORD *)(v5 + 360) = 0;
      *(_QWORD *)(v5 + 368) = 0;
      *(_QWORD *)(v5 + 376) = 0;
      *(_QWORD *)(v5 + 384) = 1;
      *(_QWORD *)(v5 + 392) = 0;
      *(_QWORD *)(v5 + 400) = 0;
      *(_DWORD *)(v5 + 408) = 0;
      *(_QWORD *)(v5 + 568) = 0;
      *(_QWORD *)(v5 + 576) = 0;
      *(_QWORD *)(v5 + 584) = 0;
      *(_DWORD *)(v5 + 592) = 0;
      *(_QWORD *)(v5 + 600) = 0;
      *(_QWORD *)(v5 + 608) = 0;
      *(_QWORD *)(v5 + 616) = 0;
      *(_QWORD *)(v5 + 624) = 0;
      *(_QWORD *)(v5 + 632) = 0;
      *(_QWORD *)v5 = off_49EDD08;
      *(_QWORD *)(v5 + 160) = &unk_49EDDC8;
    }
    sub_1611190((__int64)(v3 + 71), v6);
    a1[2] = v3;
    v3[163] = 0;
    v3[164] = 0;
    *v3 = &unk_49ED910;
    result = &unk_49EDA08;
    v3[20] = &unk_49ED9C8;
    v3[71] = &unk_49EDA08;
  }
  else
  {
    a1[2] = 0;
    v4 = 0;
  }
  v3[22] = v4;
  return result;
}
