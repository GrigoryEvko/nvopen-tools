// Function: sub_16FFCD0
// Address: 0x16ffcd0
//
void __fastcall sub_16FFCD0(_QWORD *a1, __int64 a2)
{
  __int64 *v3; // rdi
  char v5; // dl
  char v6; // al
  __int16 v7; // dx
  _BYTE *v8; // rsi
  __int64 v9; // rdx
  unsigned __int64 v10; // r14
  __int64 *v11; // rbx
  __int64 v12; // r14
  __int64 i; // r12

  v3 = a1 + 7;
  v5 = *(_BYTE *)(a2 + 8);
  *((_WORD *)v3 - 28) = *(_WORD *)a2 & 0x1FF | *(_WORD *)(v3 - 7) & 0xFE00;
  *((_DWORD *)v3 - 13) = *(_DWORD *)(a2 + 4);
  v6 = v5 & 0x1F | *(_BYTE *)(v3 - 6) & 0xE0;
  v7 = *(_WORD *)(a2 + 16);
  *((_BYTE *)v3 - 48) = v6;
  *((_DWORD *)v3 - 11) = *(_DWORD *)(a2 + 12);
  *((_WORD *)v3 - 20) = v7 & 0x3FFF | *(_WORD *)(v3 - 5) & 0xC000;
  *((_DWORD *)v3 - 9) = *(_DWORD *)(a2 + 20);
  *(v3 - 4) = *(_QWORD *)(a2 + 24);
  *(v3 - 3) = *(_QWORD *)(a2 + 32);
  *(v3 - 2) = *(_QWORD *)(a2 + 40);
  *((_WORD *)v3 - 4) = *(_WORD *)(a2 + 48) & 0x3FFF | *(_WORD *)(v3 - 1) & 0xC000;
  *((_DWORD *)v3 - 1) = *(_DWORD *)(a2 + 52);
  a1[7] = a1 + 9;
  sub_16FF8E0(v3, *(_BYTE **)(a2 + 56), *(_QWORD *)(a2 + 56) + *(_QWORD *)(a2 + 64));
  a1[11] = a1 + 13;
  v8 = *(_BYTE **)(a2 + 88);
  sub_16FF8E0(a1 + 11, v8, (__int64)&v8[*(_QWORD *)(a2 + 96)]);
  v10 = *(_QWORD *)(a2 + 128) - *(_QWORD *)(a2 + 120);
  a1[15] = 0;
  a1[16] = 0;
  a1[17] = 0;
  if ( v10 )
  {
    if ( v10 > 0x7FFFFFFFFFFFFFE0LL )
      sub_4261EA(a1 + 11, v8, v9);
    v11 = (__int64 *)sub_22077B0(v10);
  }
  else
  {
    v10 = 0;
    v11 = 0;
  }
  a1[15] = v11;
  a1[16] = v11;
  a1[17] = (char *)v11 + v10;
  v12 = *(_QWORD *)(a2 + 128);
  for ( i = *(_QWORD *)(a2 + 120); v12 != i; v11 += 4 )
  {
    if ( v11 )
    {
      *v11 = (__int64)(v11 + 2);
      sub_16FF8E0(v11, *(_BYTE **)i, *(_QWORD *)i + *(_QWORD *)(i + 8));
    }
    i += 32;
  }
  a1[16] = v11;
}
