// Function: sub_3238440
// Address: 0x3238440
//
void __fastcall sub_3238440(__int64 a1, __int64 a2, int a3, __int64 a4, __int64 a5, __int64 a6)
{
  int v6; // eax
  __int64 v10; // rdi
  __int64 v11; // rsi
  int v12; // eax
  unsigned int v13; // eax
  __int64 v14; // rdi
  bool v15; // [rsp-2Dh] [rbp-2Dh] BYREF
  unsigned int v16; // [rsp-2Ch] [rbp-2Ch] BYREF

  v6 = *(_DWORD *)(a1 + 3764);
  if ( v6 != 1 && *(_WORD *)(a2 + 36) != 74 && a5 && (v6 == 2 || a3 == 3 || !a3) )
  {
    v10 = a1 + 3080;
    if ( *(_BYTE *)(a1 + 3769) )
      v10 = a1 + 3776;
    v11 = sub_3247180(v10 + 176, *(_QWORD *)(a1 + 8), a4, a5);
    v12 = *(_DWORD *)(a1 + 3764);
    if ( v12 == 3 )
    {
      v13 = *(_DWORD *)(a2 + 72);
      v14 = *(_QWORD *)(a1 + 5384);
      v15 = *(_WORD *)(a2 + 36) == 65;
      v16 = v13;
      sub_32376B0(v14, v11, a6, &v16, (unsigned __int8 *)&v15);
    }
    else if ( v12 <= 3 )
    {
      if ( v12 == 2 )
      {
        sub_32382F0(a1 + 6016, v11, a6);
      }
      else if ( v12 >= 0 )
      {
        BUG();
      }
    }
  }
}
