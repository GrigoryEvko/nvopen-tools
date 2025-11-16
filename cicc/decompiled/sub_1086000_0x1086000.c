// Function: sub_1086000
// Address: 0x1086000
//
char __fastcall sub_1086000(__int64 a1, _QWORD *a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rax
  __int64 v7; // r12
  __int64 v8; // rax
  __int64 v9; // r12

  *(_QWORD *)(a1 + 8) = a1 + 24;
  *(_QWORD *)(a1 + 24) = a1 + 40;
  *(_WORD *)(a1 + 80) = 0;
  *(_QWORD *)(a1 + 88) = a1 + 104;
  *(_QWORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 32) = 0;
  *(_BYTE *)(a1 + 40) = 0;
  *(_QWORD *)(a1 + 56) = 0;
  *(_QWORD *)(a1 + 64) = 0;
  *(_QWORD *)(a1 + 72) = 0;
  *(_QWORD *)(a1 + 96) = 0;
  *(_QWORD *)a1 = &unk_49E6198;
  *(_QWORD *)(a1 + 104) = *a2;
  *a2 = 0;
  v6 = sub_22077B0(248);
  v7 = v6;
  if ( v6 )
    sub_1085DF0(v6, a1, a3, 1);
  *(_QWORD *)(a1 + 112) = v7;
  v8 = sub_22077B0(248);
  v9 = v8;
  if ( v8 )
    LOBYTE(v8) = sub_1085DF0(v8, a1, a4, 2);
  *(_QWORD *)(a1 + 120) = v9;
  *(_BYTE *)(a1 + 128) = 0;
  return v8;
}
