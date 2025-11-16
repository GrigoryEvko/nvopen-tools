// Function: sub_32175A0
// Address: 0x32175a0
//
void __fastcall sub_32175A0(_QWORD *a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rcx
  __int64 v4; // rsi
  __int64 v5; // rdx
  __int64 v6; // r8
  __int64 v7; // rdx
  __int64 v8; // [rsp+8h] [rbp-8h] BYREF

  v2 = a2;
  v3 = a1[5];
  v8 = a2;
  v4 = a1[4];
  v5 = (v3 - v4) >> 5;
  v6 = (v3 - v4) >> 3;
  if ( v5 <= 0 )
  {
LABEL_11:
    switch ( v6 )
    {
      case 2LL:
        v2 = v8;
        break;
      case 3LL:
        v2 = v8;
        if ( *(_QWORD *)v4 == v8 )
          goto LABEL_8;
        v4 += 8;
        break;
      case 1LL:
        v2 = v8;
LABEL_23:
        if ( *(_QWORD *)v4 == v2 )
          goto LABEL_8;
LABEL_14:
        v4 = v3;
        if ( v3 != a1[6] )
        {
          if ( !v3 )
          {
LABEL_19:
            a1[5] = v3 + 8;
            return;
          }
          v2 = v8;
LABEL_18:
          *(_QWORD *)v4 = v2;
          v3 = a1[5];
          goto LABEL_19;
        }
LABEL_34:
        sub_2E7A890((__int64)(a1 + 4), (_BYTE *)v4, &v8);
        return;
      default:
        goto LABEL_14;
    }
    if ( *(_QWORD *)v4 == v2 )
      goto LABEL_8;
    v4 += 8;
    goto LABEL_23;
  }
  v7 = v4 + 32 * v5;
  while ( *(_QWORD *)v4 != v2 )
  {
    if ( v2 == *(_QWORD *)(v4 + 8) )
    {
      v4 += 8;
      if ( v3 != v4 )
        return;
      goto LABEL_17;
    }
    if ( v2 == *(_QWORD *)(v4 + 16) )
    {
      v4 += 16;
      if ( v3 != v4 )
        return;
      goto LABEL_17;
    }
    if ( v2 == *(_QWORD *)(v4 + 24) )
    {
      v4 += 24;
      if ( v3 != v4 )
        return;
      goto LABEL_17;
    }
    v4 += 32;
    if ( v4 == v7 )
    {
      v6 = (v3 - v4) >> 3;
      goto LABEL_11;
    }
  }
LABEL_8:
  if ( v3 == v4 )
  {
LABEL_17:
    if ( v4 != a1[6] )
      goto LABEL_18;
    goto LABEL_34;
  }
}
