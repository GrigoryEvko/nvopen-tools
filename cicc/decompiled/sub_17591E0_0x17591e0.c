// Function: sub_17591E0
// Address: 0x17591e0
//
__int64 __fastcall sub_17591E0(__int64 a1, __int64 a2, _QWORD *a3, __int64 a4)
{
  __int64 v4; // rax
  __int64 v7; // r14
  _QWORD *v8; // rax

  v4 = *(_QWORD *)(a2 + 40);
  if ( !v4 )
    return 0;
  if ( a3[5] == v4 && a4 != v4 )
  {
    v7 = *(_QWORD *)(a2 + 8);
    if ( !v7 )
      return 1;
    while ( 1 )
    {
      v8 = sub_1648700(v7);
      if ( a3 != v8 && !sub_15CC8F0(*(_QWORD *)(a1 + 2656), a4, v8[5]) )
        break;
      v7 = *(_QWORD *)(v7 + 8);
      if ( !v7 )
        return 1;
    }
  }
  return 0;
}
