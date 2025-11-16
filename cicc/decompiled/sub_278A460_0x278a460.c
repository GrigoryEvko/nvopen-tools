// Function: sub_278A460
// Address: 0x278a460
//
__int64 __fastcall sub_278A460(__int64 a1, __int64 a2)
{
  int v2; // eax
  __int64 v3; // rdx
  __int64 v4; // rdx
  __int64 v5; // rax
  __int64 v6; // rax
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rdx
  __int64 v14; // rdx
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rax
  __int64 v18; // rax
  __int64 result; // rax

  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  *(_DWORD *)(a1 + 24) = 0;
  v2 = *(_DWORD *)(a2 + 24);
  v3 = *(_QWORD *)(a2 + 8);
  ++*(_QWORD *)a2;
  *(_QWORD *)(a2 + 8) = 0;
  *(_DWORD *)(a2 + 24) = 0;
  *(_QWORD *)(a1 + 8) = v3;
  v4 = *(_QWORD *)(a2 + 16);
  *(_DWORD *)(a1 + 24) = v2;
  *(_QWORD *)(a2 + 16) = 0;
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 40) = 0;
  *(_DWORD *)(a1 + 56) = 0;
  v5 = *(_QWORD *)(a2 + 40);
  *(_QWORD *)(a1 + 16) = v4;
  *(_QWORD *)(a1 + 40) = v5;
  LODWORD(v5) = *(_DWORD *)(a2 + 48);
  ++*(_QWORD *)(a2 + 32);
  *(_DWORD *)(a1 + 48) = v5;
  LODWORD(v5) = *(_DWORD *)(a2 + 52);
  *(_QWORD *)a1 = 1;
  *(_DWORD *)(a1 + 52) = v5;
  LODWORD(v5) = *(_DWORD *)(a2 + 56);
  *(_QWORD *)(a1 + 32) = 1;
  *(_DWORD *)(a1 + 56) = v5;
  *(_QWORD *)(a2 + 40) = 0;
  *(_QWORD *)(a2 + 48) = 0;
  *(_DWORD *)(a2 + 56) = 0;
  *(_DWORD *)(a1 + 64) = *(_DWORD *)(a2 + 64);
  v6 = *(_QWORD *)(a2 + 72);
  *(_QWORD *)(a2 + 72) = 0;
  *(_QWORD *)(a1 + 72) = v6;
  v7 = *(_QWORD *)(a2 + 80);
  *(_QWORD *)(a2 + 80) = 0;
  *(_QWORD *)(a1 + 80) = v7;
  v8 = *(_QWORD *)(a2 + 88);
  *(_QWORD *)(a2 + 88) = 0;
  *(_QWORD *)(a1 + 88) = v8;
  v9 = *(_QWORD *)(a2 + 96);
  *(_QWORD *)(a2 + 96) = 0;
  *(_QWORD *)(a1 + 96) = v9;
  v10 = *(_QWORD *)(a2 + 104);
  *(_QWORD *)(a2 + 104) = 0;
  *(_QWORD *)(a1 + 104) = v10;
  v11 = *(_QWORD *)(a2 + 112);
  *(_QWORD *)(a2 + 112) = 0;
  *(_QWORD *)(a1 + 112) = v11;
  *(_QWORD *)(a1 + 128) = 0;
  *(_QWORD *)(a1 + 136) = 0;
  *(_DWORD *)(a1 + 144) = 0;
  v12 = *(_QWORD *)(a2 + 128);
  LODWORD(v11) = *(_DWORD *)(a2 + 144);
  ++*(_QWORD *)(a2 + 120);
  *(_QWORD *)(a1 + 128) = v12;
  v13 = *(_QWORD *)(a2 + 136);
  *(_DWORD *)(a1 + 144) = v11;
  *(_QWORD *)(a1 + 136) = v13;
  *(_QWORD *)(a1 + 120) = 1;
  *(_QWORD *)(a2 + 128) = 0;
  *(_QWORD *)(a2 + 136) = 0;
  *(_DWORD *)(a2 + 144) = 0;
  *(_QWORD *)(a1 + 160) = 0;
  *(_QWORD *)(a1 + 168) = 0;
  *(_DWORD *)(a1 + 176) = 0;
  LODWORD(v11) = *(_DWORD *)(a2 + 176);
  v14 = *(_QWORD *)(a2 + 160);
  ++*(_QWORD *)(a2 + 152);
  *(_DWORD *)(a1 + 176) = v11;
  v15 = *(_QWORD *)(a2 + 184);
  *(_QWORD *)(a1 + 160) = v14;
  v16 = *(_QWORD *)(a2 + 168);
  *(_QWORD *)(a1 + 184) = v15;
  v17 = *(_QWORD *)(a2 + 192);
  *(_QWORD *)(a1 + 152) = 1;
  *(_QWORD *)(a1 + 192) = v17;
  v18 = *(_QWORD *)(a2 + 200);
  *(_QWORD *)(a1 + 168) = v16;
  *(_QWORD *)(a1 + 200) = v18;
  result = *(unsigned int *)(a2 + 208);
  *(_QWORD *)(a2 + 160) = 0;
  *(_QWORD *)(a2 + 168) = 0;
  *(_DWORD *)(a2 + 176) = 0;
  *(_DWORD *)(a1 + 208) = result;
  return result;
}
