// Function: sub_62F640
// Address: 0x62f640
//
__int64 __fastcall sub_62F640(__int64 a1, __int64 a2)
{
  __int64 i; // rax
  __int64 j; // rdx
  char v4; // al
  __int64 **v5; // r12
  __int64 v7; // rcx
  _QWORD *v8; // rax

  for ( i = a1; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  j = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)i + 96LL) + 8LL);
  if ( j )
  {
    v4 = *(_BYTE *)(j + 80);
    if ( v4 != 17 )
      goto LABEL_15;
    for ( j = *(_QWORD *)(j + 88); j; j = *(_QWORD *)(j + 8) )
    {
      v4 = *(_BYTE *)(j + 80);
LABEL_15:
      v7 = *(_QWORD *)(j + 88);
      if ( v4 == 20 )
      {
        v8 = *(_QWORD **)(v7 + 168);
        if ( v8 )
        {
          while ( *(_QWORD *)(v8[3] + 88LL) != a2 )
          {
            v8 = (_QWORD *)*v8;
            if ( !v8 )
              goto LABEL_13;
          }
          return 1;
        }
      }
      else if ( a2 == v7 )
      {
        return 1;
      }
LABEL_13:
      ;
    }
  }
  v5 = *(__int64 ***)(*(_QWORD *)(*(_QWORD *)(a1 + 168) + 152LL) + 176LL);
  if ( v5 )
  {
    while ( ((_BYTE)v5[5] & 4) == 0 || !(unsigned int)sub_62F640(v5[6], a2) )
    {
      v5 = (__int64 **)*v5;
      if ( !v5 )
        return 0;
    }
    return 1;
  }
  return 0;
}
