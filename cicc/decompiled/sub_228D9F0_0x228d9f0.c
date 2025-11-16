// Function: sub_228D9F0
// Address: 0x228d9f0
//
__int64 __fastcall sub_228D9F0(__int64 a1, __int64 *a2)
{
  __int64 v2; // rcx
  __int64 result; // rax
  __int64 v4; // r14
  __int64 v5; // r13
  __int64 v6; // r12

  v2 = a2[1];
  result = *(unsigned __int16 *)(*a2 + 24);
  if ( (_WORD)result == 3 )
  {
    if ( *(_WORD *)(v2 + 24) != 3 )
      return result;
    goto LABEL_5;
  }
  if ( (_WORD)result == 4 && *(_WORD *)(v2 + 24) == 4 )
  {
LABEL_5:
    v4 = *(_QWORD *)(*a2 + 32);
    v5 = *(_QWORD *)(v2 + 32);
    v6 = sub_D95540(v4);
    result = sub_D95540(v5);
    if ( v6 == result )
    {
      *a2 = v4;
      a2[1] = v5;
    }
  }
  return result;
}
