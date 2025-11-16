// Function: sub_1145160
// Address: 0x1145160
//
__int64 __fastcall sub_1145160(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  char v4; // r12
  __int64 v5; // rdx
  __int64 v6; // rcx
  __int64 v7; // r8
  __int64 v8; // r9
  _DWORD *v9; // rax
  __int64 v10[3]; // [rsp+8h] [rbp-18h] BYREF

  v2 = *(_QWORD *)(a2 + 24);
  if ( *(_BYTE *)v2 == 82
    && (v10[0] = *(_QWORD *)(a2 + 24), (*(_WORD *)(v2 + 2) & 0x3Fu) - 32 <= 1)
    && *(unsigned __int8 **)(a1 + 8) == sub_98ACB0(*(unsigned __int8 **)a2, 6u) )
  {
    v4 = sub_BD2910(a2);
    v9 = (_DWORD *)sub_1144E20(a1 + 24, v10, v5, v6, v7, v8);
    *v9 |= 1 << v4;
    return 1;
  }
  else
  {
    *(_BYTE *)(a1 + 16) = 1;
    return 0;
  }
}
