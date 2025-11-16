// Function: sub_1F405D0
// Address: 0x1f405d0
//
__int64 __fastcall sub_1F405D0(__int64 a1)
{
  _BYTE *v1; // rdx
  char v2; // cl
  _BYTE *v3; // rax
  char *v4; // rdx
  __int64 v5; // rax
  __int64 v6; // rax

  memset((void *)(a1 + 2422), 0, 0x7459u);
  memset((void *)(a1 + 32208), 0, 0x6752u);
  memset((void *)(a1 + 58658), 0, 0x33A9u);
  *(_QWORD *)(a1 + 71883) = 0;
  *(_QWORD *)(a1 + 72450) = 0;
  memset(
    (void *)((a1 + 71891) & 0xFFFFFFFFFFFFFFF8LL),
    0,
    8LL * (((unsigned int)a1 - (((_DWORD)a1 + 71891) & 0xFFFFFFF8) + 72458) >> 3));
  *(_QWORD *)(a1 + 72460) = 0;
  *(_QWORD *)(a1 + 73892) = 0;
  memset(
    (void *)((a1 + 72468) & 0xFFFFFFFFFFFFFFF8LL),
    0,
    8LL * (((unsigned int)a1 - (((_DWORD)a1 + 72468) & 0xFFFFFFF8) + 73900) >> 3));
  *(_QWORD *)(a1 + 120) = 0;
  *(_QWORD *)(a1 + 1032) = 0;
  v1 = (_BYTE *)(a1 + 71889);
  memset(
    (void *)((a1 + 128) & 0xFFFFFFFFFFFFFFF8LL),
    0,
    8LL * (((unsigned int)a1 - (((_DWORD)a1 + 128) & 0xFFFFFFF8) + 1040) >> 3));
  *(_BYTE *)(a1 + 74047) = 0;
  v2 = 1;
  v3 = (_BYTE *)(a1 + 2745);
  *(_OWORD *)(a1 + 74015) = 0;
  *(_OWORD *)(a1 + 74031) = 0;
  while ( 1 )
  {
    *v1 = 34;
    v1[1] = 34;
    v1[2] = 34;
    v1[3] = 34;
    v3[158] = 2;
    v3[38] = 2;
    v3[43] = 2;
    v3[116] = 2;
    v3[117] = 2;
    v3[118] = 2;
    v3[119] = 2;
    v3[36] = 2;
    v3[50] = 2;
    v3[51] = 2;
    v3[52] = 2;
    v3[53] = 2;
    v3[57] = 2;
    v3[6] = 2;
    v3[8] = 2;
    v3[7] = 2;
    v3[9] = 2;
    v3[10] = 2;
    v3[11] = 2;
    v3[4] = 2;
    v3[5] = 2;
    v3[74] = 2;
    *v3 = 2;
    v3[2] = 2;
    v3[1] = 2;
    v3[3] = 2;
    v3[69] = 2;
    v3[68] = 2;
    v3[67] = 2;
    v3[114] = 2;
    v3[103] = 2;
    if ( (unsigned __int8)(v2 - 14) > 0x5Fu )
      break;
    v3[37] = 2;
    v3[85] = 2;
    v3[86] = 2;
    v3[87] = 2;
    v3[179] = 2;
LABEL_2:
    v1 += 5;
    v3 += 259;
    ++v2;
  }
  v3[179] = 2;
  if ( v2 != 114 )
    goto LABEL_2;
  *(_BYTE *)(a1 + 2898) = 2;
  v4 = (char *)&unk_42F3040;
  v5 = 9;
  *(_BYTE *)(a1 + 4187) = 2;
  *(_BYTE *)(a1 + 4505) = 2;
  *(_BYTE *)(a1 + 4764) = 2;
  *(_BYTE *)(a1 + 5023) = 2;
  *(_BYTE *)(a1 + 5282) = 2;
  *(_BYTE *)(a1 + 5541) = 2;
  while ( 1 )
  {
    ++v4;
    v6 = a1 + 259 * v5;
    *(_QWORD *)(v6 + 2591) = 0x202020202020202LL;
    *(_WORD *)(v6 + 2599) = 514;
    *(_BYTE *)(v6 + 2601) = 2;
    if ( v4 == "\" pass is not registered." )
      break;
    v5 = (unsigned __int8)*v4;
  }
  *(_WORD *)(a1 + 2896) = 514;
  return 514;
}
