// Function: sub_AA4D50
// Address: 0xaa4d50
//
void __fastcall sub_AA4D50(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v8; // rax
  char v9; // al

  v8 = sub_BCB130(a2);
  sub_BD35F0(a1, v8, 23);
  *(_QWORD *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 32) = 0;
  *(_DWORD *)(a1 + 44) = -1;
  v9 = qword_4F80F48[8];
  *(_QWORD *)(a1 + 64) = 0;
  *(_BYTE *)(a1 + 40) = v9;
  *(_QWORD *)(a1 + 56) = a1 + 48;
  *(_QWORD *)(a1 + 72) = 0;
  *(_QWORD *)(a1 + 48) = (a1 + 48) | 4;
  if ( a4 )
  {
    sub_AA4C60(a1, a4, a5);
    *(_QWORD *)(a1 + 64) = a1;
    sub_BD6B50(a1, a3);
    sub_AA4C30(a1, *(_BYTE *)(a4 + 128));
  }
  else
  {
    *(_QWORD *)(a1 + 64) = a1;
    sub_BD6B50(a1, a3);
  }
}
