// Function: sub_3237830
// Address: 0x3237830
//
void __fastcall sub_3237830(__int64 a1, __int64 a2, int a3, __int64 a4, __int64 a5, __int64 a6, __int64 a7)
{
  int v7; // eax
  __int64 v11; // rdi
  __int64 v12; // rsi
  int v13; // eax
  unsigned int v14; // eax
  __int64 v15; // rdi
  bool v16; // [rsp+Bh] [rbp-25h] BYREF
  unsigned int v17[9]; // [rsp+Ch] [rbp-24h] BYREF

  v7 = *(_DWORD *)(a1 + 3764);
  if ( v7 != 1 && *(_WORD *)(a2 + 36) != 74 && a6 && (v7 == 2 || a3 == 3 || !a3) )
  {
    v11 = a1 + 3080;
    if ( *(_BYTE *)(a1 + 3769) )
      v11 = a1 + 3776;
    v12 = sub_3247180(v11 + 176, *(_QWORD *)(a1 + 8), a5, a6);
    v13 = *(_DWORD *)(a1 + 3764);
    if ( v13 == 3 )
    {
      v14 = *(_DWORD *)(a2 + 72);
      v15 = *(_QWORD *)(a1 + 5384);
      v16 = *(_WORD *)(a2 + 36) == 65;
      v17[0] = v14;
      sub_32376B0(v15, v12, a7, v17, (unsigned __int8 *)&v16);
    }
    else if ( v13 <= 3 )
    {
      if ( v13 == 2 )
      {
        sub_3237560(a4, v12, a7);
      }
      else if ( v13 >= 0 )
      {
        BUG();
      }
    }
  }
}
