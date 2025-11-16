// Function: sub_32600F0
// Address: 0x32600f0
//
_QWORD *__fastcall sub_32600F0(_QWORD *a1, __int64 a2)
{
  signed __int64 v2; // rax
  _QWORD *v3; // r12
  _QWORD *v4; // rbx

  v2 = 0xCCCCCCCCCCCCCCCDLL * ((a2 - (__int64)a1) >> 3);
  v3 = a1;
  if ( v2 >> 2 <= 0 )
  {
LABEL_17:
    if ( v2 != 2 )
    {
      if ( v2 != 3 )
      {
        if ( v2 != 1 )
          return (_QWORD *)a2;
LABEL_25:
        if ( *(_DWORD *)(*v3 + 24LL) != 51 && !(unsigned __int8)sub_33CA6D0() )
          return v3;
        return (_QWORD *)a2;
      }
      if ( *(_DWORD *)(*v3 + 24LL) != 51 && !(unsigned __int8)sub_33CA6D0() )
        return v3;
      v3 += 5;
    }
    if ( *(_DWORD *)(*v3 + 24LL) != 51 && !(unsigned __int8)sub_33CA6D0() )
      return v3;
    v3 += 5;
    goto LABEL_25;
  }
  v4 = &a1[20 * (v2 >> 2)];
  while ( *(_DWORD *)(*v3 + 24LL) == 51 || (unsigned __int8)sub_33CA6D0() )
  {
    if ( *(_DWORD *)(v3[5] + 24LL) != 51 && !(unsigned __int8)sub_33CA6D0() )
      return v3 + 5;
    if ( *(_DWORD *)(v3[10] + 24LL) != 51 && !(unsigned __int8)sub_33CA6D0() )
    {
      v3 += 10;
      return v3;
    }
    if ( *(_DWORD *)(v3[15] + 24LL) != 51 && !(unsigned __int8)sub_33CA6D0() )
    {
      v3 += 15;
      return v3;
    }
    v3 += 20;
    if ( v4 == v3 )
    {
      v2 = 0xCCCCCCCCCCCCCCCDLL * ((a2 - (__int64)v3) >> 3);
      goto LABEL_17;
    }
  }
  return v3;
}
