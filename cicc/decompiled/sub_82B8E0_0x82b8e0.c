// Function: sub_82B8E0
// Address: 0x82b8e0
//
__int64 __fastcall sub_82B8E0(__int64 a1, __int64 a2, __int64 a3, __int64 *a4)
{
  __int64 v5; // r13
  char v7; // al
  __int64 v8; // rdx
  __int64 v9; // rax
  __int64 result; // rax

  v5 = a1;
  v7 = *(_BYTE *)(a1 + 80);
  if ( v7 == 16 )
  {
    v5 = **(_QWORD **)(a1 + 88);
    v7 = *(_BYTE *)(v5 + 80);
  }
  if ( v7 == 24 )
    v5 = *(_QWORD *)(v5 + 88);
  v8 = (__int64)qword_4D03C68;
  if ( qword_4D03C68 )
    qword_4D03C68 = (_QWORD *)*qword_4D03C68;
  else
    v8 = sub_823970(152);
  *(_QWORD *)v8 = 0;
  *(_QWORD *)(v8 + 144) = 0;
  memset(
    (void *)((v8 + 8) & 0xFFFFFFFFFFFFFFF8LL),
    0,
    8LL * (((unsigned int)v8 - (((_DWORD)v8 + 8) & 0xFFFFFFF8) + 152) >> 3));
  *(_QWORD *)(v8 + 8) = a1;
  v9 = *(_QWORD *)(v5 + 88);
  if ( *(_BYTE *)(v5 + 80) == 20 )
    v9 = *(_QWORD *)(v9 + 176);
  if ( (*(_BYTE *)(v9 + 194) & 0x40) != 0 )
    *(_BYTE *)(v8 + 145) |= 0x20u;
  *(_QWORD *)(v8 + 24) = a2;
  *(_QWORD *)(v8 + 120) = a3;
  result = *a4;
  *(_QWORD *)v8 = *a4;
  *a4 = v8;
  return result;
}
