// Function: sub_97F4E0
// Address: 0x97f4e0
//
void __fastcall sub_97F4E0(__int64 a1, __int64 a2)
{
  int v2; // eax
  __int64 v3; // rdx
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // rdx
  __int64 v9; // rcx

  *(_QWORD *)(a1 + 144) = 0;
  *(_QWORD *)(a1 + 152) = 0;
  *(_DWORD *)(a1 + 160) = 0;
  v2 = *(_DWORD *)(a2 + 160);
  v3 = *(_QWORD *)(a2 + 144);
  ++*(_QWORD *)(a2 + 136);
  *(_DWORD *)(a1 + 160) = v2;
  v4 = *(_QWORD *)(a2 + 168);
  *(_QWORD *)(a1 + 144) = v3;
  v5 = *(_QWORD *)(a2 + 152);
  *(_QWORD *)(a2 + 144) = 0;
  *(_QWORD *)(a2 + 152) = 0;
  *(_DWORD *)(a2 + 160) = 0;
  *(_QWORD *)(a1 + 152) = v5;
  *(_QWORD *)(a1 + 136) = 1;
  *(_QWORD *)(a1 + 168) = v4;
  *(_QWORD *)(a1 + 176) = 0;
  *(_QWORD *)(a1 + 184) = 0;
  *(_QWORD *)(a1 + 192) = 0;
  *(_QWORD *)(a1 + 200) = 0;
  *(_QWORD *)(a1 + 208) = 0;
  *(_QWORD *)(a1 + 216) = 0;
  memmove((void *)a1, (const void *)a2, 0x83u);
  sub_97E2C0(a1 + 176, (const __m128i **)(a2 + 176), v6, v7);
  sub_97E2C0(a1 + 200, (const __m128i **)(a2 + 200), v8, v9);
}
