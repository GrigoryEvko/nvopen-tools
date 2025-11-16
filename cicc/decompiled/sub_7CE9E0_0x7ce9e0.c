// Function: sub_7CE9E0
// Address: 0x7ce9e0
//
__int64 __fastcall sub_7CE9E0(__int64 a1, __int64 a2, __int64 a3, char a4, _QWORD *a5)
{
  char v5; // bl
  __int64 v6; // r13
  _BYTE *v7; // r12
  char v8; // cl

  v5 = a4 & 1;
  v6 = sub_87EBB0(2, a1);
  v7 = sub_724D80(12);
  sub_7249B0((__int64)v7, 3);
  *(_QWORD *)(v6 + 88) = v7;
  v8 = v5 | v7[177] & 0xFE;
  *((_QWORD *)v7 + 16) = dword_4D03B80;
  v7[177] = v8;
  v7[200] = *(_BYTE *)(a1 + 72);
  *a5 = v7;
  sub_877F50(v7, v6, 0xFFFFFFFFLL);
  if ( a2 )
  {
    sub_877E20(v6, v7, a2);
  }
  else if ( a3 )
  {
    sub_877E90(v6, v7);
  }
  *(_BYTE *)(v6 + 84) |= 1u;
  *(_DWORD *)(v6 + 40) = *(_DWORD *)(*(_QWORD *)(unk_4D03FF0 + 8LL) + 24LL);
  return v6;
}
