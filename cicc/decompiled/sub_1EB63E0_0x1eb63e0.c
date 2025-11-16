// Function: sub_1EB63E0
// Address: 0x1eb63e0
//
__int64 __fastcall sub_1EB63E0(__int64 a1, __int64 a2, unsigned int a3, unsigned __int16 a4)
{
  __int64 v5; // rbx
  unsigned int v6; // r15d
  int v8; // esi
  __int64 v9; // r8
  unsigned __int64 v10; // r9
  int v11; // edx
  unsigned __int8 v12; // al
  int v13; // edx
  unsigned int v15; // eax
  unsigned __int8 v16; // [rsp+Fh] [rbp-31h]

  v5 = *(_QWORD *)(a2 + 32) + 40LL * a3;
  v6 = ((*(_BYTE *)(v5 + 3) & 0x40) != 0) & (*(_BYTE *)(v5 + 3) >> 4);
  if ( ((*(_DWORD *)v5 >> 8) & 0xFFF) != 0 )
  {
    v8 = 0;
    if ( a4 )
      v8 = sub_38D6F10(*(_QWORD *)(a1 + 248) + 8LL, a4, (*(_DWORD *)v5 >> 8) & 0xFFF);
    sub_1E310D0(v5, v8);
    sub_1E31360(v5, 1);
    v11 = HIBYTE(*(_DWORD *)v5);
    *(_DWORD *)v5 &= 0xFFF000FF;
    v12 = (unsigned __int8)v11 >> 6;
    v13 = v11 & 0x10;
    if ( (v12 & (v13 == 0)) != 0 )
    {
      v16 = v12 & (v13 == 0);
      LOBYTE(v9) = v16;
      sub_1E1AFE0(a2, a4, *(_QWORD **)(a1 + 248), 1, v9, v10);
      return v16;
    }
    else if ( (_BYTE)v13 && (*(_BYTE *)(v5 + 4) & 1) != 0 )
    {
      sub_1E1B830(a2, a4, *(_QWORD *)(a1 + 248));
    }
  }
  else
  {
    sub_1E310D0(v5, a4);
    sub_1E31360(v5, 1);
    v15 = (*(_BYTE *)(v5 + 3) & 0x40) != 0;
    LOBYTE(v15) = ((*(_BYTE *)(v5 + 3) >> 4) ^ 1) & v15;
    if ( (_BYTE)v15 )
      return v15;
  }
  return v6;
}
