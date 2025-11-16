// Function: sub_1299230
// Address: 0x1299230
//
__int64 __fastcall sub_1299230(_QWORD **a1, __int64 a2)
{
  __int64 v3; // rsi
  __int64 i; // rax
  __int64 v5; // rax
  unsigned __int8 v6; // dl

  v3 = sub_1297B70(a1, *(_QWORD *)(a2 + 152), (*(_BYTE *)(a2 + 199) & 8) != 0);
  for ( i = *(_QWORD *)(a2 + 152); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  v5 = *(_QWORD *)(i + 168);
  v6 = 0;
  if ( v5 )
    v6 = *(_BYTE *)(v5 + 16) & 1;
  return sub_1299060(a1, v3, v6, (_DWORD *)(a2 + 64));
}
