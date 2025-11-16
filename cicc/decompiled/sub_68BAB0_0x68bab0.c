// Function: sub_68BAB0
// Address: 0x68bab0
//
__int64 __fastcall sub_68BAB0(__int64 a1, _BYTE *a2, _DWORD *a3, __int64 a4, __int64 a5)
{
  __int64 v7; // rsi
  unsigned int v8; // r13d
  __int64 v10; // rax
  char i; // dl

  v7 = *(_QWORD *)a2;
  if ( v7 == a1 || (v8 = sub_8D97D0(a1, v7, 0, a4, a5)) != 0 )
  {
    if ( (unsigned int)sub_6E53E0(5, 1211, a3) )
      sub_684B30(0x4BBu, a3);
    return 1;
  }
  else if ( a2[16] )
  {
    v10 = *(_QWORD *)a2;
    for ( i = *(_BYTE *)(*(_QWORD *)a2 + 140LL); i == 12; i = *(_BYTE *)(v10 + 140) )
      v10 = *(_QWORD *)(v10 + 160);
    if ( i && (unsigned int)sub_6E5430() )
      sub_685360(0x77u, a3, a1);
  }
  return v8;
}
