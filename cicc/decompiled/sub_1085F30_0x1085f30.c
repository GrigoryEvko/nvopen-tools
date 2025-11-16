// Function: sub_1085F30
// Address: 0x1085f30
//
char __fastcall sub_1085F30(__int64 a1, _QWORD *a2, __int64 a3)
{
  __int64 v4; // rax
  __int64 v5; // r12

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
  v4 = sub_22077B0(248);
  v5 = v4;
  if ( v4 )
    LOBYTE(v4) = sub_1085DF0(v4, a1, a3, 0);
  *(_QWORD *)(a1 + 112) = v5;
  *(_QWORD *)(a1 + 120) = 0;
  *(_BYTE *)(a1 + 128) = 0;
  return v4;
}
