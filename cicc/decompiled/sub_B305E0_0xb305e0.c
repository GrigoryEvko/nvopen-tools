// Function: sub_B305E0
// Address: 0xb305e0
//
__int64 __fastcall sub_B305E0(__int64 a1, _QWORD *a2, unsigned int a3, char a4, __int64 a5, __int64 a6, __int64 a7)
{
  __int64 v9; // rax
  int v10; // eax
  __int64 result; // rax
  bool v12; // zf
  __int64 v13; // rax
  __int64 v14; // rcx
  __int64 v15; // rax

  v9 = sub_BCE3C0(*a2, a3);
  sub_BD35F0(a1, v9, 2);
  v10 = *(_DWORD *)(a1 + 4);
  *(_QWORD *)(a1 + 24) = a2;
  *(_QWORD *)(a1 + 40) = 0;
  *(_DWORD *)(a1 + 4) = v10 & 0x38000000 | 1;
  *(_QWORD *)(a1 + 32) = *(_QWORD *)(a1 + 32) & 0xFFFE0000LL | a4 & 0xF;
  if ( (a4 & 0xFu) - 7 <= 1 )
    *(_BYTE *)(a1 + 33) |= 0x40u;
  result = sub_BD6B50(a1, a5);
  *(_WORD *)(a1 + 34) &= 1u;
  v12 = *(_QWORD *)(a1 - 32) == 0;
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 56) = 0;
  *(_QWORD *)(a1 + 64) = 0;
  if ( !v12 )
  {
    result = *(_QWORD *)(a1 - 24);
    **(_QWORD **)(a1 - 16) = result;
    if ( result )
      *(_QWORD *)(result + 16) = *(_QWORD *)(a1 - 16);
  }
  *(_QWORD *)(a1 - 32) = a6;
  if ( a6 )
  {
    v13 = *(_QWORD *)(a6 + 16);
    *(_QWORD *)(a1 - 24) = v13;
    if ( v13 )
      *(_QWORD *)(v13 + 16) = a1 - 24;
    result = a1 - 32;
    *(_QWORD *)(a1 - 16) = a6 + 16;
    *(_QWORD *)(a6 + 16) = a1 - 32;
  }
  if ( a7 )
  {
    sub_BA86C0(a7 + 56, a1);
    v14 = *(_QWORD *)(a7 + 56);
    v15 = *(_QWORD *)(a1 + 56);
    *(_QWORD *)(a1 + 64) = a7 + 56;
    v14 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)(a1 + 56) = v14 | v15 & 7;
    *(_QWORD *)(v14 + 8) = a1 + 56;
    result = *(_QWORD *)(a7 + 56) & 7LL;
    *(_QWORD *)(a7 + 56) = result | (a1 + 56);
  }
  return result;
}
