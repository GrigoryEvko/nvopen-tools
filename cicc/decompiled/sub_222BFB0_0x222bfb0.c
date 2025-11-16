// Function: sub_222BFB0
// Address: 0x222bfb0
//
__int64 __fastcall sub_222BFB0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  int v4; // r13d
  __off64_t v6; // r8
  __int64 v8; // rax
  __int64 v9; // rax

  v4 = a3;
  if ( !(unsigned __int8)sub_222BE90(a1, a2, a3, a4) )
    return -1;
  v6 = sub_2207F40((FILE **)(a1 + 104), a2, v4);
  if ( v6 != -1 )
  {
    *(_QWORD *)(a1 + 132) = a4;
    *(_WORD *)(a1 + 169) = 0;
    v8 = *(_QWORD *)(a1 + 208);
    *(_QWORD *)(a1 + 40) = 0;
    *(_QWORD *)(a1 + 232) = v8;
    *(_QWORD *)(a1 + 224) = v8;
    v9 = *(_QWORD *)(a1 + 152);
    *(_QWORD *)(a1 + 32) = 0;
    *(_QWORD *)(a1 + 8) = v9;
    *(_QWORD *)(a1 + 16) = v9;
    *(_QWORD *)(a1 + 24) = v9;
    *(_QWORD *)(a1 + 48) = 0;
  }
  return v6;
}
