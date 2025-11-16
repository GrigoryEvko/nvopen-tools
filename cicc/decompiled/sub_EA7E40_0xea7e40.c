// Function: sub_EA7E40
// Address: 0xea7e40
//
__int64 __fastcall sub_EA7E40(__int64 a1, int *a2)
{
  __int64 *v4; // rdi
  int v5; // edx
  _BYTE *v6; // rsi
  __int64 v7; // rdx
  unsigned __int64 v8; // r12
  __int64 *v9; // rbx
  __int64 v10; // r14
  __int64 i; // r12
  int v12; // edx
  __int64 result; // rax

  v4 = (__int64 *)(a1 + 32);
  v5 = *a2;
  BYTE1(v5) = BYTE1(*a2) & 0x3F;
  *((_DWORD *)v4 - 8) = v5 | *(_DWORD *)(v4 - 4) & 0xC000;
  *((_WORD *)v4 - 14) = *((_WORD *)a2 + 2);
  *(v4 - 3) = *((_QWORD *)a2 + 1);
  *(v4 - 2) = *((_QWORD *)a2 + 2);
  *(v4 - 1) = *((_QWORD *)a2 + 3);
  *(_QWORD *)(a1 + 32) = a1 + 48;
  sub_EA2980(v4, *((_BYTE **)a2 + 4), *((_QWORD *)a2 + 4) + *((_QWORD *)a2 + 5));
  *(_QWORD *)(a1 + 64) = a1 + 80;
  sub_EA2980((__int64 *)(a1 + 64), *((_BYTE **)a2 + 8), *((_QWORD *)a2 + 8) + *((_QWORD *)a2 + 9));
  *(_QWORD *)(a1 + 96) = a1 + 112;
  sub_EA2980((__int64 *)(a1 + 96), *((_BYTE **)a2 + 12), *((_QWORD *)a2 + 12) + *((_QWORD *)a2 + 13));
  *(_QWORD *)(a1 + 128) = a1 + 144;
  sub_EA2980((__int64 *)(a1 + 128), *((_BYTE **)a2 + 16), *((_QWORD *)a2 + 16) + *((_QWORD *)a2 + 17));
  *(_QWORD *)(a1 + 160) = a1 + 176;
  sub_EA2980((__int64 *)(a1 + 160), *((_BYTE **)a2 + 20), *((_QWORD *)a2 + 20) + *((_QWORD *)a2 + 21));
  *(_QWORD *)(a1 + 192) = a1 + 208;
  v6 = (_BYTE *)*((_QWORD *)a2 + 24);
  sub_EA2980((__int64 *)(a1 + 192), v6, (__int64)&v6[*((_QWORD *)a2 + 25)]);
  v8 = *((_QWORD *)a2 + 29) - *((_QWORD *)a2 + 28);
  *(_QWORD *)(a1 + 224) = 0;
  *(_QWORD *)(a1 + 232) = 0;
  *(_QWORD *)(a1 + 240) = 0;
  if ( v8 )
  {
    if ( v8 > 0x7FFFFFFFFFFFFFE0LL )
      sub_4261EA(a1 + 192, v6, v7);
    v9 = (__int64 *)sub_22077B0(v8);
  }
  else
  {
    v8 = 0;
    v9 = 0;
  }
  *(_QWORD *)(a1 + 224) = v9;
  *(_QWORD *)(a1 + 232) = v9;
  *(_QWORD *)(a1 + 240) = (char *)v9 + v8;
  v10 = *((_QWORD *)a2 + 29);
  for ( i = *((_QWORD *)a2 + 28); v10 != i; v9 += 4 )
  {
    if ( v9 )
    {
      *v9 = (__int64)(v9 + 2);
      sub_EA2980(v9, *(_BYTE **)i, *(_QWORD *)i + *(_QWORD *)(i + 8));
    }
    i += 32;
  }
  *(_QWORD *)(a1 + 232) = v9;
  v12 = a2[62] & 3;
  result = v12 | *(_BYTE *)(a1 + 248) & 0xFCu;
  *(_BYTE *)(a1 + 248) = v12 | *(_BYTE *)(a1 + 248) & 0xFC;
  return result;
}
