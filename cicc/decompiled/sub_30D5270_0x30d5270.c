// Function: sub_30D5270
// Address: 0x30d5270
//
__int64 __fastcall sub_30D5270(__int64 a1)
{
  int v2; // eax
  __int64 v3; // rsi
  __int64 v4; // rdi
  unsigned __int8 v5; // al
  __int64 v6; // rdi
  int v7; // r12d
  int v8; // eax
  int v9; // edx

  v2 = sub_30D4FE0(*(__int64 **)(a1 + 8), *(unsigned __int8 **)(a1 + 96), *(_QWORD *)(a1 + 80));
  v3 = *(_QWORD *)(a1 + 72);
  *(_DWORD *)(a1 + 720) -= v2;
  v4 = *(_QWORD *)(a1 + 96);
  *(_DWORD *)(a1 + 724) = ((*(_WORD *)(v3 + 2) >> 4) & 0x3FF) == 9;
  v5 = sub_30D14D0(v4, v3);
  v6 = *(_QWORD *)(a1 + 8);
  *(_DWORD *)(a1 + 728) = v5;
  v7 = sub_DF94D0(v6);
  *(_DWORD *)(a1 + 760) += sub_DF9470(*(_QWORD *)(a1 + 8));
  LODWORD(v3) = *(_DWORD *)(a1 + 760) * sub_DF93B0(*(_QWORD *)(a1 + 8));
  v8 = 50 * (int)v3 / 100;
  *(_DWORD *)(a1 + 756) = v8;
  v9 = (int)v3 * v7 / 100;
  *(_DWORD *)(a1 + 752) = v9;
  *(_DWORD *)(a1 + 760) = v3 + v9 + v8;
  return 0;
}
