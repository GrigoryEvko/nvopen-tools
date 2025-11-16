// Function: sub_ABE8C0
// Address: 0xabe8c0
//
__int64 __fastcall sub_ABE8C0(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  _QWORD *v3; // rdx
  __int64 result; // rax
  __int64 v5; // rdx
  _BYTE *v6; // rax
  _BYTE *v7; // rax
  __int64 v8; // rdx
  char *v9; // rsi

  v2 = a2;
  if ( sub_AAF760(a1) )
  {
    v3 = *(_QWORD **)(a2 + 32);
    if ( *(_QWORD *)(a2 + 24) - (_QWORD)v3 > 7u )
    {
      *v3 = 0x7465732D6C6C7566LL;
      *(_QWORD *)(a2 + 32) += 8LL;
      return 0x7465732D6C6C7566LL;
    }
    v8 = 8;
    v9 = "full-set";
  }
  else if ( sub_AAF7D0(a1) )
  {
    v5 = *(_QWORD *)(a2 + 32);
    if ( (unsigned __int64)(*(_QWORD *)(a2 + 24) - v5) > 8 )
    {
      *(_BYTE *)(v5 + 8) = 116;
      *(_QWORD *)v5 = 0x65732D7974706D65LL;
      *(_QWORD *)(a2 + 32) += 9LL;
      return 0x65732D7974706D65LL;
    }
    v8 = 9;
    v9 = "empty-set";
  }
  else
  {
    v6 = *(_BYTE **)(a2 + 32);
    if ( *(_BYTE **)(a2 + 24) == v6 )
    {
      v2 = sub_CB6200(a2, "[", 1);
    }
    else
    {
      *v6 = 91;
      ++*(_QWORD *)(a2 + 32);
    }
    sub_C49420(a1, v2, 1);
    v7 = *(_BYTE **)(v2 + 32);
    if ( *(_BYTE **)(v2 + 24) == v7 )
    {
      v2 = sub_CB6200(v2, ",", 1);
    }
    else
    {
      *v7 = 44;
      ++*(_QWORD *)(v2 + 32);
    }
    sub_C49420(a1 + 16, v2, 1);
    result = *(_QWORD *)(v2 + 32);
    if ( *(_QWORD *)(v2 + 24) != result )
    {
      *(_BYTE *)result = 41;
      ++*(_QWORD *)(v2 + 32);
      return result;
    }
    v8 = 1;
    v9 = ")";
  }
  return sub_CB6200(v2, v9, v8);
}
