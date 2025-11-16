// Function: sub_25F6310
// Address: 0x25f6310
//
__int64 __fastcall sub_25F6310(__int64 a1, __int64 a2)
{
  __int64 v4; // rax
  __int64 v5; // rsi
  __int64 v6; // rdx
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rax
  __int64 result; // rax

  *(_DWORD *)a1 = *(_DWORD *)a2;
  *(_DWORD *)(a1 + 4) = *(_DWORD *)(a2 + 4);
  *(_QWORD *)(a1 + 8) = *(_QWORD *)(a2 + 8);
  v4 = *(_QWORD *)(a2 + 16);
  v5 = *(unsigned int *)(a1 + 48);
  *(_QWORD *)(a1 + 16) = v4;
  sub_C7D6A0(*(_QWORD *)(a1 + 32), 16 * v5, 8);
  *(_QWORD *)(a1 + 40) = 0;
  *(_QWORD *)(a1 + 32) = 0;
  *(_DWORD *)(a1 + 48) = 0;
  ++*(_QWORD *)(a1 + 24);
  v6 = *(_QWORD *)(a2 + 32);
  ++*(_QWORD *)(a2 + 24);
  v7 = *(_QWORD *)(a1 + 32);
  *(_QWORD *)(a1 + 32) = v6;
  LODWORD(v6) = *(_DWORD *)(a2 + 40);
  *(_QWORD *)(a2 + 32) = v7;
  LODWORD(v7) = *(_DWORD *)(a1 + 40);
  *(_DWORD *)(a1 + 40) = v6;
  LODWORD(v6) = *(_DWORD *)(a2 + 44);
  *(_DWORD *)(a2 + 40) = v7;
  LODWORD(v7) = *(_DWORD *)(a1 + 44);
  *(_DWORD *)(a1 + 44) = v6;
  LODWORD(v6) = *(_DWORD *)(a2 + 48);
  *(_DWORD *)(a2 + 44) = v7;
  LODWORD(v7) = *(_DWORD *)(a1 + 48);
  *(_DWORD *)(a1 + 48) = v6;
  *(_DWORD *)(a2 + 48) = v7;
  sub_C7D6A0(*(_QWORD *)(a1 + 64), 16LL * *(unsigned int *)(a1 + 80), 8);
  *(_QWORD *)(a1 + 72) = 0;
  *(_QWORD *)(a1 + 64) = 0;
  *(_DWORD *)(a1 + 80) = 0;
  ++*(_QWORD *)(a1 + 56);
  v8 = *(_QWORD *)(a2 + 64);
  ++*(_QWORD *)(a2 + 56);
  v9 = *(_QWORD *)(a1 + 64);
  *(_QWORD *)(a1 + 64) = v8;
  LODWORD(v8) = *(_DWORD *)(a2 + 72);
  *(_QWORD *)(a2 + 64) = v9;
  LODWORD(v9) = *(_DWORD *)(a1 + 72);
  *(_DWORD *)(a1 + 72) = v8;
  LODWORD(v8) = *(_DWORD *)(a2 + 76);
  *(_DWORD *)(a2 + 72) = v9;
  LODWORD(v9) = *(_DWORD *)(a1 + 76);
  *(_DWORD *)(a1 + 76) = v8;
  LODWORD(v8) = *(_DWORD *)(a2 + 80);
  *(_DWORD *)(a2 + 76) = v9;
  LODWORD(v9) = *(_DWORD *)(a1 + 80);
  *(_DWORD *)(a1 + 80) = v8;
  *(_DWORD *)(a2 + 80) = v9;
  sub_C7D6A0(*(_QWORD *)(a1 + 96), 8LL * *(unsigned int *)(a1 + 112), 4);
  *(_QWORD *)(a1 + 104) = 0;
  *(_QWORD *)(a1 + 96) = 0;
  *(_DWORD *)(a1 + 112) = 0;
  ++*(_QWORD *)(a1 + 88);
  v10 = *(_QWORD *)(a2 + 96);
  ++*(_QWORD *)(a2 + 88);
  v11 = *(_QWORD *)(a1 + 96);
  *(_QWORD *)(a1 + 96) = v10;
  LODWORD(v10) = *(_DWORD *)(a2 + 104);
  *(_QWORD *)(a2 + 96) = v11;
  LODWORD(v11) = *(_DWORD *)(a1 + 104);
  *(_DWORD *)(a1 + 104) = v10;
  LODWORD(v10) = *(_DWORD *)(a2 + 108);
  *(_DWORD *)(a2 + 104) = v11;
  LODWORD(v11) = *(_DWORD *)(a1 + 108);
  *(_DWORD *)(a1 + 108) = v10;
  LODWORD(v10) = *(_DWORD *)(a2 + 112);
  *(_DWORD *)(a2 + 108) = v11;
  LODWORD(v11) = *(_DWORD *)(a1 + 112);
  *(_DWORD *)(a1 + 112) = v10;
  *(_DWORD *)(a2 + 112) = v11;
  sub_C7D6A0(*(_QWORD *)(a1 + 128), 8LL * *(unsigned int *)(a1 + 144), 4);
  ++*(_QWORD *)(a1 + 120);
  *(_QWORD *)(a1 + 136) = 0;
  *(_QWORD *)(a1 + 128) = 0;
  *(_DWORD *)(a1 + 144) = 0;
  v12 = *(_QWORD *)(a2 + 128);
  ++*(_QWORD *)(a2 + 120);
  v13 = *(_QWORD *)(a1 + 128);
  *(_QWORD *)(a1 + 128) = v12;
  LODWORD(v12) = *(_DWORD *)(a2 + 136);
  *(_QWORD *)(a2 + 128) = v13;
  LODWORD(v13) = *(_DWORD *)(a1 + 136);
  *(_DWORD *)(a1 + 136) = v12;
  LODWORD(v12) = *(_DWORD *)(a2 + 140);
  *(_DWORD *)(a2 + 136) = v13;
  LODWORD(v13) = *(_DWORD *)(a1 + 140);
  *(_DWORD *)(a1 + 140) = v12;
  LODWORD(v12) = *(_DWORD *)(a2 + 144);
  *(_DWORD *)(a2 + 140) = v13;
  result = *(unsigned int *)(a1 + 144);
  *(_DWORD *)(a1 + 144) = v12;
  *(_DWORD *)(a2 + 144) = result;
  return result;
}
