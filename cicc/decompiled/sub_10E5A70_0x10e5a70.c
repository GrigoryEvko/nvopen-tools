// Function: sub_10E5A70
// Address: 0x10e5a70
//
bool __fastcall sub_10E5A70(__int64 *a1, __int64 *a2)
{
  _QWORD *v2; // rcx
  __int64 v3; // rdx
  __int64 v4; // rsi
  _QWORD *v5; // rax
  _QWORD *v6; // rax

  v2 = (_QWORD *)a1[1];
  v3 = *a1;
  if ( v2 == (_QWORD *)*a1 )
  {
    v5 = (_QWORD *)a1[1];
    return v5 != v2;
  }
  else
  {
    v4 = *a2;
    while ( 1 )
    {
      while ( 1 )
      {
        v6 = (_QWORD *)(v3 & 0xFFFFFFFFFFFFFFF8LL);
        if ( (v3 & 4) == 0 )
          break;
        if ( v4 == *(_QWORD *)(*v6 + 136LL) )
          return v3 != (_QWORD)v2;
        v3 = (unsigned __int64)(v6 + 1) | 4;
        v5 = (_QWORD *)v3;
        if ( v2 == (_QWORD *)v3 )
          return v5 != v2;
      }
      if ( v4 == v6[17] )
        break;
      v5 = v6 + 18;
      v3 = (__int64)v5;
      if ( v2 == v5 )
        return v5 != v2;
    }
    return v3 != (_QWORD)v2;
  }
}
