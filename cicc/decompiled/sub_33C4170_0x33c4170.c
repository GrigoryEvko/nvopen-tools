// Function: sub_33C4170
// Address: 0x33c4170
//
void __fastcall sub_33C4170(__int64 a1, _BYTE *a2)
{
  __int64 v4; // rdi
  int v5; // eax
  __int64 v6; // rsi
  int v7; // ecx
  int v8; // r8d
  unsigned int v9; // eax
  _BYTE *v10; // rdx
  int v11; // eax

  if ( *a2 == 22 || *a2 > 0x1Cu )
  {
    v4 = *(_QWORD *)(a1 + 960);
    v5 = *(_DWORD *)(v4 + 144);
    v6 = *(_QWORD *)(v4 + 128);
    if ( v5 )
    {
      v7 = v5 - 1;
      v8 = 1;
      v9 = (v5 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v10 = *(_BYTE **)(v6 + 16LL * v9);
      if ( a2 == v10 )
        return;
      while ( v10 != (_BYTE *)-4096LL )
      {
        v9 = v7 & (v8 + v9);
        v10 = *(_BYTE **)(v6 + 16LL * v9);
        if ( a2 == v10 )
          return;
        ++v8;
      }
    }
    v11 = sub_374D810(v4, a2);
    sub_33BF9C0(a1, (__int64)a2, v11, 215);
  }
}
