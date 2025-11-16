// Function: sub_1592EE0
// Address: 0x1592ee0
//
__int64 __fastcall sub_1592EE0(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  _QWORD *v3; // rdx
  __int64 result; // rax
  __int64 v5; // rdx
  _BYTE *v6; // rax
  _BYTE *v7; // rax

  v2 = a2;
  if ( sub_158A0B0(a1) )
  {
    v3 = *(_QWORD **)(a2 + 24);
    if ( *(_QWORD *)(a2 + 16) - (_QWORD)v3 <= 7u )
    {
      return sub_16E7EE0(a2, "full-set", 8);
    }
    else
    {
      *v3 = 0x7465732D6C6C7566LL;
      *(_QWORD *)(a2 + 24) += 8LL;
      return 0x7465732D6C6C7566LL;
    }
  }
  else if ( sub_158A120(a1) )
  {
    v5 = *(_QWORD *)(a2 + 24);
    if ( (unsigned __int64)(*(_QWORD *)(a2 + 16) - v5) <= 8 )
    {
      return sub_16E7EE0(a2, "empty-set", 9);
    }
    else
    {
      *(_BYTE *)(v5 + 8) = 116;
      *(_QWORD *)v5 = 0x65732D7974706D65LL;
      *(_QWORD *)(a2 + 24) += 9LL;
      return 0x65732D7974706D65LL;
    }
  }
  else
  {
    v6 = *(_BYTE **)(a2 + 24);
    if ( *(_BYTE **)(a2 + 16) == v6 )
    {
      v2 = sub_16E7EE0(a2, "[", 1);
    }
    else
    {
      *v6 = 91;
      ++*(_QWORD *)(a2 + 24);
    }
    sub_16A95F0(a1, v2, 1);
    v7 = *(_BYTE **)(v2 + 24);
    if ( *(_BYTE **)(v2 + 16) == v7 )
    {
      v2 = sub_16E7EE0(v2, ",", 1);
    }
    else
    {
      *v7 = 44;
      ++*(_QWORD *)(v2 + 24);
    }
    sub_16A95F0(a1 + 16, v2, 1);
    result = *(_QWORD *)(v2 + 24);
    if ( *(_QWORD *)(v2 + 16) == result )
    {
      return sub_16E7EE0(v2, ")", 1);
    }
    else
    {
      *(_BYTE *)result = 41;
      ++*(_QWORD *)(v2 + 24);
    }
  }
  return result;
}
