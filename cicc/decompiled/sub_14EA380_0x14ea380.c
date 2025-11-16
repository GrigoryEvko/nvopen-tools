// Function: sub_14EA380
// Address: 0x14ea380
//
__int64 __fastcall sub_14EA380(__int64 a1, __int64 a2, unsigned int a3, int a4, __int64 a5)
{
  __int64 v5; // rax
  __int64 v6; // rsi
  __int64 v8; // rax

  if ( a3 == *(_DWORD *)(a2 + 8) )
    return 0;
  v5 = *(_QWORD *)(*(_QWORD *)a2 + 8LL * a3);
  v6 = (unsigned int)v5;
  if ( *(_BYTE *)(a1 + 1656) )
    v6 = (unsigned int)(a4 - v5);
  if ( !a5 || *(_BYTE *)(a5 + 8) != 8 )
    return sub_1522F40(a1 + 552, v6);
  v8 = sub_1521F50(a1 + 608, v6);
  return sub_1628DA0(*(_QWORD *)a5, v8);
}
