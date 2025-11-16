// Function: sub_5C8460
// Address: 0x5c8460
//
__int64 __fastcall sub_5C8460(__int64 a1, __int64 a2)
{
  __int64 v3; // r13
  __int64 v4; // rax
  __int64 v5; // rsi
  char *v7; // rax

  v3 = *(_QWORD *)(a1 + 32);
  v4 = sub_692720(*(_QWORD *)(v3 + 40));
  *(_QWORD *)(v3 + 40) = v4;
  if ( !*(_BYTE *)(v4 + 24) )
    goto LABEL_7;
  if ( *(_BYTE *)(a2 + 140) != 7 || (v5 = *(_QWORD *)(a1 + 48), (*(_BYTE *)(v5 + 122) & 2) != 0) )
  {
    v7 = sub_5C79F0(a1);
    sub_684B10(1835, a1 + 56, v7);
LABEL_7:
    *(_BYTE *)(a1 + 8) = 0;
    return a2;
  }
  *(_BYTE *)(*(_QWORD *)(a2 + 168) + 20LL) |= 2u;
  if ( *(_BYTE *)(a1 + 8) == 19 )
  {
    sub_643E40(sub_5C8070, v5, 1);
    return a2;
  }
  return a2;
}
