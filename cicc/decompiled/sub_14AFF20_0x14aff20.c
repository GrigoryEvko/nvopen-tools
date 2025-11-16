// Function: sub_14AFF20
// Address: 0x14aff20
//
__int64 __fastcall sub_14AFF20(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rbx
  __int64 v4; // r15
  __int64 v6; // rbx
  __int64 v7; // rdx
  __int64 v8; // rax
  __int64 v9; // rdx

  if ( a3 )
  {
    if ( !(unsigned __int8)sub_15CCEE0(a3, a1, a2) )
    {
      if ( *(_QWORD *)(a1 + 40) == *(_QWORD *)(a2 + 40) )
        goto LABEL_4;
      return 0;
    }
    return 1;
  }
  v6 = *(_QWORD *)(a1 + 40);
  if ( v6 == sub_157F0B0(*(_QWORD *)(a2 + 40)) )
    return 1;
  v7 = *(_QWORD *)(a1 + 40);
  if ( v7 != *(_QWORD *)(a2 + 40) )
    return 0;
  v8 = *(_QWORD *)(a1 + 32);
  v9 = v7 + 40;
  if ( v9 != v8 )
  {
    while ( !v8 || a2 != v8 - 24 )
    {
      v8 = *(_QWORD *)(v8 + 8);
      if ( v9 == v8 )
        goto LABEL_4;
    }
    return 1;
  }
LABEL_4:
  v3 = *(_QWORD *)(a2 + 32);
  if ( a1 + 24 != v3 )
  {
    while ( 1 )
    {
      v4 = 0;
      if ( v3 )
        v4 = v3 - 24;
      if ( !(unsigned __int8)sub_14AF470(v4, 0, 0, 0) && !sub_14AAB80(v4) )
        break;
      v3 = *(_QWORD *)(v3 + 8);
      if ( a1 + 24 == v3 )
        return (unsigned int)sub_14AF8E0(a1, a2) ^ 1;
    }
    return 0;
  }
  return (unsigned int)sub_14AF8E0(a1, a2) ^ 1;
}
