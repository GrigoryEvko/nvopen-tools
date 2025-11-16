// Function: sub_1443660
// Address: 0x1443660
//
__int64 __fastcall sub_1443660(_QWORD *a1, __int64 a2)
{
  __int64 result; // rax
  int v3; // r12d
  __int64 v4; // r13
  unsigned int i; // r15d
  __int64 v6; // r14
  __int64 v7; // r12
  unsigned __int64 v8; // [rsp+8h] [rbp-48h]
  __int64 v9; // [rsp+18h] [rbp-38h]

  if ( !(unsigned __int8)sub_1443560(a1, a2) )
    sub_16BD130("Broken region found: enumerated BB not in region!", 1);
  v8 = *a1 & 0xFFFFFFFFFFFFFFF8LL;
  result = sub_157EBA0(a2);
  if ( result )
  {
    v9 = a1[4];
    v3 = sub_15F4D60(result);
    result = sub_157EBA0(a2);
    v4 = result;
    if ( v3 )
    {
      for ( i = 0; i != v3; ++i )
      {
        v6 = sub_15F4DF0(v4, i);
        result = sub_1443560(a1, v6);
        if ( v9 != v6 && (_BYTE)result != 1 )
          sub_16BD130("Broken region found: edges leaving the region must go to the exit node!", 1);
      }
    }
  }
  if ( a2 != v8 )
  {
    result = a2;
    v7 = *(_QWORD *)(a2 + 8);
    if ( v7 )
    {
      while ( 1 )
      {
        result = sub_1648700(v7);
        if ( (unsigned __int8)(*(_BYTE *)(result + 16) - 25) <= 9u )
          break;
        v7 = *(_QWORD *)(v7 + 8);
        if ( !v7 )
          return result;
      }
LABEL_14:
      result = sub_1443560(a1, *(_QWORD *)(result + 40));
      if ( !(_BYTE)result )
        sub_16BD130("Broken region found: edges entering the region must go to the entry node!", 1);
      while ( 1 )
      {
        v7 = *(_QWORD *)(v7 + 8);
        if ( !v7 )
          break;
        result = sub_1648700(v7);
        if ( (unsigned __int8)(*(_BYTE *)(result + 16) - 25) <= 9u )
          goto LABEL_14;
      }
    }
  }
  return result;
}
