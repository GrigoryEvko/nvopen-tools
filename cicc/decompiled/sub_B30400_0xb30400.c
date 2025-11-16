// Function: sub_B30400
// Address: 0xb30400
//
void __fastcall sub_B30400(__int64 a1, _QWORD *a2, unsigned int a3, char a4, __int64 a5, __int64 a6, __int64 a7)
{
  __int64 v9; // rax
  int v10; // eax
  __int64 v11; // rcx
  __int64 v12; // rax

  v9 = sub_BCE3C0(*a2, a3);
  sub_BD35F0(a1, v9, 1);
  v10 = *(_DWORD *)(a1 + 4);
  *(_QWORD *)(a1 + 24) = a2;
  *(_QWORD *)(a1 + 40) = 0;
  *(_DWORD *)(a1 + 4) = v10 & 0x38000000 | 1;
  *(_QWORD *)(a1 + 32) = *(_QWORD *)(a1 + 32) & 0xFFFE0000LL | a4 & 0xF;
  if ( (a4 & 0xFu) - 7 <= 1 )
    *(_BYTE *)(a1 + 33) |= 0x40u;
  sub_BD6B50(a1, a5);
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 56) = 0;
  sub_B303B0(a1, a6);
  if ( a7 )
  {
    sub_BA8640(a7 + 40, a1);
    v11 = *(_QWORD *)(a7 + 40);
    v12 = *(_QWORD *)(a1 + 48);
    *(_QWORD *)(a1 + 56) = a7 + 40;
    v11 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)(a1 + 48) = v11 | v12 & 7;
    *(_QWORD *)(v11 + 8) = a1 + 48;
    *(_QWORD *)(a7 + 40) = *(_QWORD *)(a7 + 40) & 7LL | (a1 + 48);
  }
}
