// Function: sub_314D600
// Address: 0x314d600
//
void __fastcall sub_314D600(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rdx
  __int64 v7; // rdx
  int v8; // eax
  __int64 v9; // rdx
  __int64 v10; // rdx
  int v11; // eax
  __int64 v12; // rdx

  *(_QWORD *)a1 = a1 + 16;
  *(_QWORD *)(a1 + 8) = 0x10000000000LL;
  v6 = *(unsigned int *)(a2 + 8);
  if ( (_DWORD)v6 )
    sub_314D4A0(a1, (char **)a2, v6, a4, a5, a6);
  *(_QWORD *)(a1 + 2072) = 0;
  *(_QWORD *)(a1 + 2080) = 0;
  *(_DWORD *)(a1 + 2088) = 0;
  v7 = *(_QWORD *)(a2 + 2072);
  v8 = *(_DWORD *)(a2 + 2088);
  ++*(_QWORD *)(a2 + 2064);
  *(_QWORD *)(a1 + 2072) = v7;
  v9 = *(_QWORD *)(a2 + 2080);
  *(_QWORD *)(a2 + 2072) = 0;
  *(_QWORD *)(a2 + 2080) = 0;
  *(_DWORD *)(a2 + 2088) = 0;
  *(_QWORD *)(a1 + 2080) = v9;
  *(_QWORD *)(a1 + 2104) = 0;
  *(_QWORD *)(a1 + 2112) = 0;
  *(_DWORD *)(a1 + 2120) = 0;
  v10 = *(_QWORD *)(a2 + 2104);
  *(_DWORD *)(a1 + 2088) = v8;
  v11 = *(_DWORD *)(a2 + 2120);
  *(_QWORD *)(a1 + 2104) = v10;
  v12 = *(_QWORD *)(a2 + 2112);
  *(_DWORD *)(a1 + 2120) = v11;
  ++*(_QWORD *)(a2 + 2096);
  *(_QWORD *)(a2 + 2104) = 0;
  *(_QWORD *)(a2 + 2112) = 0;
  *(_DWORD *)(a2 + 2120) = 0;
  *(_QWORD *)(a1 + 2128) = a1 + 2144;
  *(_QWORD *)(a1 + 2064) = 1;
  *(_QWORD *)(a1 + 2096) = 1;
  *(_QWORD *)(a1 + 2112) = v12;
  *(_QWORD *)(a1 + 2136) = 0x1000000000LL;
  if ( *(_DWORD *)(a2 + 2136) )
    sub_314D4A0(a1 + 2128, (char **)(a2 + 2128), v12, a4, a5, a6);
}
