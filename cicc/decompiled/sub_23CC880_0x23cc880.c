// Function: sub_23CC880
// Address: 0x23cc880
//
__int64 __fastcall sub_23CC880(_QWORD *a1, __int64 a2)
{
  __int64 v2; // r10
  __int64 v3; // r9
  __int64 v4; // rcx
  __int64 v5; // rdx
  __int64 v6; // rax
  unsigned __int64 v7; // r8
  signed __int64 v8; // r11
  __int64 v9; // rdi

  v2 = a1[6];
  v3 = a1[9];
  v4 = a1[5];
  v5 = a1[4];
  v6 = a1[2];
  v7 = 0xCCCCCCCCCCCCCCCDLL * ((v2 - a1[7]) >> 3);
  v8 = 0xCCCCCCCCCCCCCCCDLL * ((v5 - v6) >> 3) + v7 + 4 * (3 * ((v3 - v4) >> 3) - 3);
  v9 = v8 >> 2;
  if ( v8 >> 2 > 0 )
  {
    while ( *(_QWORD *)(v6 + 32) != a2 )
    {
      v6 += 40;
      if ( v5 == v6 )
      {
        v6 = *(_QWORD *)(v4 + 8);
        v4 += 8;
        v5 = v6 + 480;
        if ( *(_QWORD *)(v6 + 32) == a2 )
          goto LABEL_15;
      }
      else if ( *(_QWORD *)(v6 + 32) == a2 )
      {
        goto LABEL_15;
      }
      v6 += 40;
      if ( v5 == v6 )
      {
        v6 = *(_QWORD *)(v4 + 8);
        v4 += 8;
        v5 = v6 + 480;
      }
      if ( *(_QWORD *)(v6 + 32) == a2 )
        break;
      v6 += 40;
      if ( v5 == v6 )
      {
        v6 = *(_QWORD *)(v4 + 8);
        v4 += 8;
        v5 = v6 + 480;
      }
      if ( *(_QWORD *)(v6 + 32) == a2 )
        break;
      v6 += 40;
      if ( v5 == v6 )
      {
        v6 = *(_QWORD *)(v4 + 8);
        v4 += 8;
        v5 = v6 + 480;
        if ( !--v9 )
        {
LABEL_18:
          v7 += 4 * (3 * ((v3 - v4) >> 3) - 3);
          v8 = v7 - 0x3333333333333333LL * ((v5 - v6) >> 3);
          goto LABEL_19;
        }
      }
      else if ( !--v9 )
      {
        goto LABEL_18;
      }
    }
    goto LABEL_15;
  }
LABEL_19:
  if ( v8 == 2 )
    goto LABEL_27;
  if ( v8 != 3 )
  {
    LODWORD(v7) = 0;
    if ( v8 != 1 )
      return (unsigned int)v7;
    goto LABEL_22;
  }
  if ( *(_QWORD *)(v6 + 32) != a2 )
  {
    v6 += 40;
    if ( v6 == v5 )
    {
      v6 = *(_QWORD *)(v4 + 8);
      v4 += 8;
      v5 = v6 + 480;
    }
LABEL_27:
    if ( *(_QWORD *)(v6 + 32) != a2 )
    {
      v6 += 40;
      if ( v6 == v5 )
        v6 = *(_QWORD *)(v4 + 8);
LABEL_22:
      LODWORD(v7) = 0;
      if ( *(_QWORD *)(v6 + 32) != a2 )
        return (unsigned int)v7;
    }
  }
LABEL_15:
  LOBYTE(v7) = v2 != v6;
  return (unsigned int)v7;
}
