// Function: sub_25FE910
// Address: 0x25fe910
//
__int64 __fastcall sub_25FE910(__int64 a1, __int64 *a2)
{
  __int64 v2; // rdx
  __int64 v3; // rax
  __int64 v4; // rcx
  __int64 v5; // rdx
  __int64 v6; // rdx
  __int64 v7; // rdx
  __int64 v8; // rdx
  __int64 v9; // rdx
  __int64 v10; // rdx
  __int64 v11; // rdx
  __int64 result; // rax
  __int64 v13; // rdx

  v2 = a2[1];
  v3 = a2[2];
  *(_QWORD *)(a1 + 24) = 1;
  v4 = *a2;
  *(_QWORD *)(a1 + 32) = 0;
  *(_QWORD *)(a1 + 40) = 0;
  *(_DWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 8) = v2;
  v5 = a2[4];
  ++a2[3];
  *(_QWORD *)(a1 + 32) = v5;
  v6 = a2[5];
  a2[4] = 0;
  a2[5] = 0;
  *(_QWORD *)(a1 + 16) = v3;
  LODWORD(v3) = *((_DWORD *)a2 + 12);
  *(_QWORD *)(a1 + 40) = v6;
  *((_DWORD *)a2 + 12) = 0;
  *(_QWORD *)(a1 + 64) = 0;
  *(_QWORD *)(a1 + 72) = 0;
  *(_DWORD *)(a1 + 80) = 0;
  v7 = a2[8];
  *(_DWORD *)(a1 + 48) = v3;
  LODWORD(v3) = *((_DWORD *)a2 + 20);
  *(_QWORD *)(a1 + 64) = v7;
  v8 = a2[9];
  ++a2[7];
  *(_QWORD *)(a1 + 72) = v8;
  *(_QWORD *)a1 = v4;
  *(_QWORD *)(a1 + 56) = 1;
  *(_DWORD *)(a1 + 80) = v3;
  a2[8] = 0;
  a2[9] = 0;
  *((_DWORD *)a2 + 20) = 0;
  *(_QWORD *)(a1 + 96) = 0;
  *(_QWORD *)(a1 + 104) = 0;
  *(_DWORD *)(a1 + 112) = 0;
  v9 = a2[12];
  LODWORD(v3) = *((_DWORD *)a2 + 28);
  ++a2[11];
  *(_QWORD *)(a1 + 96) = v9;
  v10 = a2[13];
  a2[12] = 0;
  a2[13] = 0;
  *((_DWORD *)a2 + 28) = 0;
  *(_QWORD *)(a1 + 104) = v10;
  *(_QWORD *)(a1 + 128) = 0;
  *(_QWORD *)(a1 + 136) = 0;
  *(_DWORD *)(a1 + 144) = 0;
  v11 = a2[16];
  *(_DWORD *)(a1 + 112) = v3;
  result = *((unsigned int *)a2 + 36);
  *(_QWORD *)(a1 + 128) = v11;
  v13 = a2[17];
  ++a2[15];
  *(_QWORD *)(a1 + 88) = 1;
  *(_QWORD *)(a1 + 120) = 1;
  *(_QWORD *)(a1 + 136) = v13;
  *(_DWORD *)(a1 + 144) = result;
  a2[16] = 0;
  a2[17] = 0;
  *((_DWORD *)a2 + 36) = 0;
  return result;
}
