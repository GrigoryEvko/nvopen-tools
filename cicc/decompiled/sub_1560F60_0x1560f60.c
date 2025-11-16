// Function: sub_1560F60
// Address: 0x1560f60
//
__int64 __fastcall sub_1560F60(__int64 a1, __int64 a2)
{
  __int64 v2; // rbx
  __int64 v3; // r14
  __int64 v4; // rax
  __int64 v5; // r14
  __int64 v6; // rax
  __int64 v7; // r14
  __int64 v8; // rax
  __int64 v9; // r14
  __int64 v10; // rax
  __int64 v11; // r13
  __int64 v12; // rax

  v2 = sub_1560310((_QWORD *)(a2 + 112), -1, 0x2Au);
  LOBYTE(v2) = v2 == sub_1560310((_QWORD *)(a1 + 112), -1, 0x2Au);
  v3 = sub_1560310((_QWORD *)(a2 + 112), -1, 0x2Du);
  v4 = sub_1560310((_QWORD *)(a1 + 112), -1, 0x2Du);
  LOBYTE(v4) = v3 == v4;
  LODWORD(v2) = v4 & v2;
  v5 = sub_1560310((_QWORD *)(a2 + 112), -1, 0x2Cu);
  v6 = sub_1560310((_QWORD *)(a1 + 112), -1, 0x2Cu);
  LOBYTE(v6) = v5 == v6;
  LODWORD(v2) = v6 & v2;
  v7 = sub_1560310((_QWORD *)(a2 + 112), -1, 0x2Bu);
  v8 = sub_1560310((_QWORD *)(a1 + 112), -1, 0x2Bu);
  LOBYTE(v8) = v7 == v8;
  LODWORD(v2) = v8 & v2;
  v9 = sub_1560310((_QWORD *)(a2 + 112), -1, 0x29u);
  v10 = sub_1560310((_QWORD *)(a1 + 112), -1, 0x29u);
  LOBYTE(v10) = v9 == v10;
  LODWORD(v2) = v10 & v2;
  v11 = sub_1560310((_QWORD *)(a2 + 112), -1, 0x2Eu);
  v12 = sub_1560310((_QWORD *)(a1 + 112), -1, 0x2Eu);
  LOBYTE(v12) = v11 == v12;
  return (unsigned int)v2 & (unsigned int)v12;
}
