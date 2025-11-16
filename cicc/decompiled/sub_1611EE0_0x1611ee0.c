// Function: sub_1611EE0
// Address: 0x1611ee0
//
_QWORD *__fastcall sub_1611EE0(_QWORD *a1)
{
  _QWORD *result; // rax
  _QWORD *v2; // rbx
  char *v3; // r13
  __int64 v4; // rax
  __int64 v5; // rsi

  *a1 = &unk_49EDE08;
  a1[1] = 0;
  result = (_QWORD *)sub_22077B0(1320);
  v2 = result;
  if ( result )
  {
    result[1] = 0;
    *((_DWORD *)result + 6) = 6;
    v3 = (char *)(result + 71);
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
    v4 = sub_22077B0(640);
    v5 = v4;
    if ( v4 )
    {
      *(_QWORD *)(v4 + 8) = 0;
      *(_QWORD *)(v4 + 16) = &unk_4F9E3B9;
      *(_QWORD *)(v4 + 80) = v4 + 64;
      *(_QWORD *)(v4 + 88) = v4 + 64;
      *(_QWORD *)(v4 + 128) = v4 + 112;
      *(_QWORD *)(v4 + 136) = v4 + 112;
      *(_QWORD *)(v4 + 184) = v4 + 200;
      v5 = v4 + 160;
      *(_QWORD *)(v4 + 416) = v4 + 432;
      *(_DWORD *)(v4 + 24) = 6;
      *(_QWORD *)(v4 + 32) = 0;
      *(_QWORD *)(v4 + 40) = 0;
      *(_QWORD *)(v4 + 48) = 0;
      *(_DWORD *)(v4 + 64) = 0;
      *(_QWORD *)(v4 + 72) = 0;
      *(_QWORD *)(v4 + 96) = 0;
      *(_DWORD *)(v4 + 112) = 0;
      *(_QWORD *)(v4 + 120) = 0;
      *(_QWORD *)(v4 + 144) = 0;
      *(_BYTE *)(v4 + 152) = 0;
      *(_QWORD *)(v4 + 168) = 0;
      *(_QWORD *)(v4 + 176) = 0;
      *(_QWORD *)(v4 + 192) = 0x1000000000LL;
      *(_QWORD *)(v4 + 424) = 0x1000000000LL;
      *(_DWORD *)(v4 + 560) = 0;
      *(_QWORD *)(v4 + 328) = 0;
      *(_QWORD *)(v4 + 336) = 0;
      *(_QWORD *)(v4 + 344) = 0;
      *(_QWORD *)(v4 + 352) = 0;
      *(_QWORD *)(v4 + 360) = 0;
      *(_QWORD *)(v4 + 368) = 0;
      *(_QWORD *)(v4 + 376) = 0;
      *(_QWORD *)(v4 + 384) = 1;
      *(_QWORD *)(v4 + 392) = 0;
      *(_QWORD *)(v4 + 400) = 0;
      *(_DWORD *)(v4 + 408) = 0;
      *(_QWORD *)(v4 + 568) = 0;
      *(_QWORD *)(v4 + 576) = 0;
      *(_QWORD *)(v4 + 584) = 0;
      *(_DWORD *)(v4 + 592) = 0;
      *(_QWORD *)(v4 + 600) = 0;
      *(_QWORD *)(v4 + 608) = 0;
      *(_QWORD *)(v4 + 616) = 0;
      *(_QWORD *)(v4 + 624) = 0;
      *(_QWORD *)(v4 + 632) = 0;
      *(_QWORD *)v4 = off_49EDD08;
      *(_QWORD *)(v4 + 160) = &unk_49EDDC8;
    }
    sub_1611190((__int64)(v2 + 71), v5);
    a1[2] = v2;
    v2[163] = 0;
    v2[164] = 0;
    *v2 = &unk_49ED910;
    result = &unk_49EDA08;
    v2[20] = &unk_49ED9C8;
    v2[71] = &unk_49EDA08;
  }
  else
  {
    a1[2] = 0;
    v3 = 0;
  }
  v2[22] = v3;
  return result;
}
