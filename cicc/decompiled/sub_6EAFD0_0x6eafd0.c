// Function: sub_6EAFD0
// Address: 0x6eafd0
//
__int64 __fastcall sub_6EAFD0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v8; // rax
  char v9; // si
  __int64 v10; // r12

  if ( dword_4D048B8 && a4 )
  {
    *(_QWORD *)(a1 + 16) = a4;
    v8 = qword_4D03C50;
    v9 = *(_BYTE *)(qword_4D03C50 + 17LL);
    if ( (v9 & 2) != 0 )
    {
      *(_BYTE *)(a4 + 193) |= 0x40u;
      v9 = *(_BYTE *)(v8 + 17);
    }
    sub_734250(a1, v9 & 1);
  }
  v10 = sub_6EAFA0(6u);
  sub_63BA50(a1, a2, a3, v10, a5);
  return v10;
}
