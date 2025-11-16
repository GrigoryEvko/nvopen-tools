// Function: sub_13A6E90
// Address: 0x13a6e90
//
__int64 __fastcall sub_13A6E90(__int64 a1, _QWORD *a2)
{
  __int64 v2; // rcx
  __int64 result; // rax
  __int64 v4; // r14
  __int64 v5; // r13
  __int64 v6; // r12

  v2 = a2[1];
  result = *(unsigned __int16 *)(*a2 + 24LL);
  if ( (_WORD)result == 2 )
  {
    if ( *(_WORD *)(v2 + 24) != 2 )
      return result;
    goto LABEL_5;
  }
  if ( (_WORD)result == 3 && *(_WORD *)(v2 + 24) == 3 )
  {
LABEL_5:
    v4 = *(_QWORD *)(*a2 + 32LL);
    v5 = *(_QWORD *)(v2 + 32);
    v6 = sub_1456040(v4);
    result = sub_1456040(v5);
    if ( v6 == result )
    {
      *a2 = v4;
      a2[1] = v5;
    }
  }
  return result;
}
