// Function: sub_6E7350
// Address: 0x6e7350
//
__int64 __fastcall sub_6E7350(__int64 a1, char a2, __int64 a3, __int64 a4)
{
  __int64 v5; // r14
  char v7; // al
  __int64 v8; // rax
  char v9; // r12
  __int64 result; // rax

  v5 = a1;
  v7 = *(_BYTE *)(a1 + 80);
  if ( v7 == 16 )
  {
    v5 = **(_QWORD **)(a1 + 88);
    v7 = *(_BYTE *)(v5 + 80);
  }
  if ( v7 == 24 )
    v5 = *(_QWORD *)(v5 + 88);
  sub_6E2E50(4, a4);
  if ( *(_BYTE *)(v5 + 80) == 8 )
  {
    *(_BYTE *)(a4 + 17) = 2;
    v8 = *(_QWORD *)(*(_QWORD *)(v5 + 88) + 120LL);
  }
  else
  {
    *(_BYTE *)(a4 + 17) = 3;
    v8 = *(_QWORD *)(*(_QWORD *)(v5 + 88) + 152LL);
  }
  *(_QWORD *)a4 = v8;
  *(_QWORD *)(a4 + 136) = a1;
  v9 = *(_BYTE *)(a4 + 18);
  *(_QWORD *)(a4 + 88) = a3;
  *(_QWORD *)(a4 + 68) = *(_QWORD *)&dword_4F063F8;
  *(_BYTE *)(a4 + 18) = ((a2 & 1) << 6) | v9 & 0xBF;
  result = qword_4F063F0;
  *(_QWORD *)(a4 + 76) = qword_4F063F0;
  return result;
}
