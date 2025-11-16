// Function: sub_AC3E90
// Address: 0xac3e90
//
__int64 __fastcall sub_AC3E90(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned int v4; // r14d
  __int64 v5; // rax
  __int64 v6; // rax
  bool v7; // zf
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // rax
  unsigned int v12; // eax
  unsigned int v13; // edx
  __int64 result; // rax

  v4 = *(_DWORD *)(*(_QWORD *)(a2 + 8) + 8LL);
  v5 = sub_B2BE50(a2);
  v6 = sub_BCE3C0(v5, v4 >> 8);
  sub_BD35F0(a1, v6, 4);
  v7 = *(_QWORD *)(a1 - 64) == 0;
  *(_DWORD *)(a1 + 4) = *(_DWORD *)(a1 + 4) & 0x38000000 | 2;
  if ( !v7 )
  {
    v8 = *(_QWORD *)(a1 - 56);
    **(_QWORD **)(a1 - 48) = v8;
    if ( v8 )
      *(_QWORD *)(v8 + 16) = *(_QWORD *)(a1 - 48);
  }
  v9 = *(_QWORD *)(a2 + 16);
  *(_QWORD *)(a1 - 64) = a2;
  *(_QWORD *)(a1 - 56) = v9;
  if ( v9 )
    *(_QWORD *)(v9 + 16) = a1 - 56;
  v7 = *(_QWORD *)(a1 - 32) == 0;
  *(_QWORD *)(a1 - 48) = a2 + 16;
  *(_QWORD *)(a2 + 16) = a1 - 64;
  if ( !v7 )
  {
    v10 = *(_QWORD *)(a1 - 24);
    **(_QWORD **)(a1 - 16) = v10;
    if ( v10 )
      *(_QWORD *)(v10 + 16) = *(_QWORD *)(a1 - 16);
  }
  *(_QWORD *)(a1 - 32) = a3;
  if ( a3 )
  {
    v11 = *(_QWORD *)(a3 + 16);
    *(_QWORD *)(a1 - 24) = v11;
    if ( v11 )
      *(_QWORD *)(v11 + 16) = a1 - 24;
    *(_QWORD *)(a1 - 16) = a3 + 16;
    *(_QWORD *)(a3 + 16) = a1 - 32;
  }
  v12 = *(unsigned __int16 *)(a3 + 2);
  v13 = v12 + 1;
  LOWORD(v12) = v12 & 0x8000;
  LOWORD(v13) = v13 & 0x7FFF;
  result = v13 | v12;
  *(_WORD *)(a3 + 2) = result;
  return result;
}
