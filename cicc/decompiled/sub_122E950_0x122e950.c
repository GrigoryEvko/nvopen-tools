// Function: sub_122E950
// Address: 0x122e950
//
__int64 __fastcall sub_122E950(__int64 a1, __int64 *a2, __int64 *a3)
{
  char v3; // r13
  __int64 v5; // rdi
  unsigned __int64 v6; // rsi
  unsigned int v7; // r12d
  __int64 v9; // rax
  __int64 v11; // rdx
  __int64 v12; // [rsp+0h] [rbp-80h] BYREF
  __int64 v13; // [rsp+8h] [rbp-78h] BYREF
  __int64 v14; // [rsp+10h] [rbp-70h] BYREF
  __int64 v15; // [rsp+18h] [rbp-68h] BYREF
  __int64 v16; // [rsp+20h] [rbp-60h] BYREF
  __int64 v17; // [rsp+28h] [rbp-58h] BYREF
  __int64 v18[4]; // [rsp+30h] [rbp-50h] BYREF
  char v19; // [rsp+50h] [rbp-30h]
  char v20; // [rsp+51h] [rbp-2Fh]

  v5 = a1 + 176;
  if ( *(_DWORD *)(a1 + 240) != 525 )
  {
    v6 = *(_QWORD *)(a1 + 232);
    v20 = 1;
    v7 = 1;
    v18[0] = (__int64)"expected debug record type here";
    v19 = 3;
    sub_11FD800(v5, v6, (__int64)v18, 1);
    return v7;
  }
  v9 = *(_QWORD *)(a1 + 256);
  v11 = *(_QWORD *)(a1 + 248);
  if ( v9 == 7 )
  {
    if ( *(_DWORD *)v11 == 1818453348 && *(_WORD *)(v11 + 4) == 29281 )
      v3 = 0;
  }
  else if ( v9 == 5 )
  {
    if ( (*(_DWORD *)v11 != 1970037110 || *(_BYTE *)(v11 + 4) != 101)
      && *(_DWORD *)v11 == 1700946284
      && *(_BYTE *)(v11 + 4) == 108 )
    {
      *(_DWORD *)(a1 + 240) = sub_1205200(v5);
      if ( !(unsigned __int8)sub_120AFE0(a1, 12, "Expected '(' here")
        && !(unsigned __int8)sub_122E7D0(a1, &v17)
        && !(unsigned __int8)sub_120AFE0(a1, 4, "Expected ',' here")
        && !(unsigned __int8)sub_122E7D0(a1, v18) )
      {
        v7 = sub_120AFE0(a1, 13, "Expected ')' here");
        if ( !(_BYTE)v7 )
        {
          *a2 = sub_B12630(v17, v18[0]);
          return v7;
        }
      }
      return 1;
    }
    if ( *(_DWORD *)v11 == 1970037110 )
      v3 = *(_BYTE *)(v11 + 4) == 101;
  }
  else if ( v9 == 6 && *(_DWORD *)v11 == 1769173857 )
  {
    v3 = 0;
    if ( *(_WORD *)(v11 + 4) == 28263 )
      v3 = 2;
  }
  *(_DWORD *)(a1 + 240) = sub_1205200(v5);
  if ( !(unsigned __int8)sub_120AFE0(a1, 12, "Expected '(' here")
    && !(unsigned __int8)sub_12254B0(a1, &v12, a3)
    && !(unsigned __int8)sub_120AFE0(a1, 4, "Expected ',' here")
    && !(unsigned __int8)sub_122E7D0(a1, &v13)
    && !(unsigned __int8)sub_120AFE0(a1, 4, "Expected ',' here")
    && !(unsigned __int8)sub_122E7D0(a1, &v14)
    && !(unsigned __int8)sub_120AFE0(a1, 4, "Expected ',' here") )
  {
    v15 = 0;
    v16 = 0;
    v17 = 0;
    if ( (v3 != 2
       || !(unsigned __int8)sub_122E7D0(a1, &v15)
       && !(unsigned __int8)sub_120AFE0(a1, 4, "Expected ',' here")
       && !(unsigned __int8)sub_12254B0(a1, &v16, a3)
       && !(unsigned __int8)sub_120AFE0(a1, 4, "Expected ',' here")
       && !(unsigned __int8)sub_122E7D0(a1, &v17)
       && !(unsigned __int8)sub_120AFE0(a1, 4, "Expected ',' here"))
      && !(unsigned __int8)sub_122E7D0(a1, v18) )
    {
      v7 = sub_120AFE0(a1, 13, "Expected ')' here");
      if ( !(_BYTE)v7 )
      {
        *a2 = sub_B12780(v3, v12, v13, v14, v15, v16, v17, v18[0]);
        return v7;
      }
    }
  }
  return 1;
}
