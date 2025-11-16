// Function: sub_10C2350
// Address: 0x10c2350
//
__int64 __fastcall sub_10C2350(__int64 a1, unsigned __int8 *a2)
{
  __int64 v2; // rbx
  unsigned __int8 *v3; // r12
  int v4; // edx
  __int64 v6; // rdi
  __int64 v7; // rdi
  __int64 v8; // rdx
  __int64 v9; // r8
  __int64 v10; // rcx
  _BYTE *v11; // rdi
  _QWORD *v12[6]; // [rsp+0h] [rbp-30h] BYREF

  v2 = *(_QWORD *)(a1 + 16);
  if ( v2 )
  {
    while ( 1 )
    {
      v3 = *(unsigned __int8 **)(v2 + 24);
      if ( v3 == a2 )
        goto LABEL_10;
      v4 = *v3;
      if ( v4 == 59 )
      {
        v12[0] = 0;
        if ( *v3 != 59
          || !(unsigned __int8)sub_995B10(v12, *((_QWORD *)v3 - 8))
          && !(unsigned __int8)sub_995B10(v12, *((_QWORD *)v3 - 4)) )
        {
          return 0;
        }
        goto LABEL_10;
      }
      if ( v4 != 86 )
      {
        if ( v4 != 31 )
          return 0;
        goto LABEL_10;
      }
      if ( (unsigned int)sub_BD2910(v2) )
        return 0;
      v6 = *((_QWORD *)v3 + 1);
      if ( (unsigned int)*(unsigned __int8 *)(v6 + 8) - 17 <= 1 )
        v6 = **(_QWORD **)(v6 + 16);
      if ( !sub_BCAC40(v6, 1) )
        goto LABEL_30;
      if ( *v3 == 57 )
        return 0;
      v7 = *((_QWORD *)v3 + 1);
      if ( *v3 == 86 && *(_QWORD *)(*((_QWORD *)v3 - 12) + 8LL) == v7 && **((_BYTE **)v3 - 4) <= 0x15u )
        break;
LABEL_19:
      if ( (unsigned int)*(unsigned __int8 *)(v7 + 8) - 17 <= 1 )
        v7 = **(_QWORD **)(v7 + 16);
      if ( sub_BCAC40(v7, 1) )
      {
        if ( *v3 == 58 )
          return 0;
        if ( *v3 == 86 )
        {
          v10 = *((_QWORD *)v3 + 1);
          if ( *(_QWORD *)(*((_QWORD *)v3 - 12) + 8LL) == v10 )
          {
            v11 = (_BYTE *)*((_QWORD *)v3 - 8);
            if ( *v11 <= 0x15u && sub_AD7A80(v11, 1, v8, v10, v9) )
              return 0;
          }
        }
      }
LABEL_10:
      v2 = *(_QWORD *)(v2 + 8);
      if ( !v2 )
        return 1;
    }
    if ( sub_AC30F0(*((_QWORD *)v3 - 4)) )
      return 0;
LABEL_30:
    v7 = *((_QWORD *)v3 + 1);
    goto LABEL_19;
  }
  return 1;
}
