// Function: sub_73E1B0
// Address: 0x73e1b0
//
_BYTE *__fastcall sub_73E1B0(__int64 a1, __int64 a2)
{
  char v2; // al
  __int64 v3; // r12
  _BYTE *result; // rax
  __int64 v5; // r13
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 v9; // r9

  v2 = *(_BYTE *)(a1 + 24);
  v3 = a1;
  if ( !v2 )
    return (_BYTE *)v3;
  if ( v2 == 1 && ((*(_BYTE *)(a1 + 27) & 2) != 0 || dword_4D03F94) && *(_BYTE *)(a1 + 56) == 3 )
    return *(_BYTE **)(a1 + 72);
  v5 = sub_731400((_QWORD **)a1);
  sub_7313A0(a1, a2, v6, v7, v8, v9);
  *(_QWORD *)(a1 + 16) = 0;
  result = sub_73DBF0(0, v5, a1);
  result[27] |= 2u;
  return result;
}
