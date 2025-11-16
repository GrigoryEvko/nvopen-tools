// Function: sub_326B690
// Address: 0x326b690
//
__int64 __fastcall sub_326B690(__int64 a1, _QWORD *a2)
{
  int v2; // eax
  _QWORD *v4; // rax
  __int64 v5; // rax
  __int64 v6; // r14
  unsigned int v7; // r13d
  __int64 v8; // rax
  __int64 v9; // rax

  v2 = *(_DWORD *)(a1 + 24);
  if ( v2 != 187 )
  {
    if ( v2 != 192 )
      return 0;
    v4 = *(_QWORD **)(a1 + 40);
    if ( *(_DWORD *)(*v4 + 24LL) != 197 )
      return 0;
    v5 = sub_33DFBC0(v4[5], v4[6], 0, 0);
    if ( !v5 )
      return 0;
    v6 = *(_QWORD *)(v5 + 96);
    v7 = *(_DWORD *)(v6 + 32);
    if ( v7 > 0x40 )
    {
      if ( v7 - (unsigned int)sub_C444A0(v6 + 24) > 0x40 )
        return 0;
      v8 = **(_QWORD **)(v6 + 24);
    }
    else
    {
      v8 = *(_QWORD *)(v6 + 24);
    }
    if ( v8 == 16 )
    {
      v9 = **(_QWORD **)(**(_QWORD **)(a1 + 40) + 40LL);
      a2[1] = v9;
      *a2 = v9;
      return 1;
    }
    return 0;
  }
  if ( !(unsigned __int8)sub_325E3E0(**(_QWORD **)(a1 + 40), (__int64)a2) )
    return 0;
  return sub_325E3E0(*(_QWORD *)(*(_QWORD *)(a1 + 40) + 40LL), (__int64)a2);
}
