// Function: sub_D09E10
// Address: 0xd09e10
//
__int64 __fastcall sub_D09E10(
        __int64 *a1,
        unsigned __int64 a2,
        __int64 a3,
        unsigned __int64 a4,
        __int64 a5,
        __int64 a6,
        char *a7,
        char *a8)
{
  __int64 *v8; // r10
  unsigned __int8 v13; // al
  unsigned __int8 v14; // dl
  int v15; // eax
  unsigned __int8 v16; // cl
  int v17; // esi
  int v18; // edx
  int v19; // eax
  __int64 result; // rax
  unsigned int v21; // eax
  int v22; // edx
  int v23; // eax
  int v24; // ecx
  char v25; // al
  unsigned __int8 v26; // r12
  __int64 *v29; // [rsp+8h] [rbp-38h]
  __int64 *v30; // [rsp+8h] [rbp-38h]

  v8 = (__int64 *)a6;
  v13 = *(_BYTE *)a2;
  if ( *(_BYTE *)a2 > 0x1Cu )
  {
    if ( v13 != 63 )
      goto LABEL_3;
LABEL_13:
    result = sub_D06280(a1, (unsigned __int8 *)a2, a3, a4, a5, a7, (__int64)a8, a6);
    v8 = (__int64 *)a6;
    if ( (_BYTE)result != 1 )
      return result;
    goto LABEL_14;
  }
  if ( v13 == 5 && *(_WORD *)(a2 + 2) == 34 )
    goto LABEL_13;
LABEL_3:
  v14 = *(_BYTE *)a4;
  if ( *(_BYTE *)a4 > 0x1Cu )
  {
    if ( v14 != 63 )
      goto LABEL_15;
    goto LABEL_5;
  }
  if ( v14 == 5 && *(_WORD *)(a4 + 2) == 34 )
  {
LABEL_5:
    v15 = sub_D06280(a1, (unsigned __int8 *)a4, a5, a2, a3, a8, (__int64)a7, a6);
    v8 = (__int64 *)a6;
    v16 = v15;
    v17 = v15 >> 9;
    v18 = ((unsigned int)v15 >> 8) & 1;
    v19 = BYTE1(v15) & 1;
    if ( v19 )
    {
      if ( v17 == -4194304 )
      {
        v17 = -4194304;
      }
      else
      {
        LOBYTE(v18) = v19;
        v17 = (-512 * v17) >> 9;
      }
    }
    if ( v16 != 1 )
      return (v17 << 9) | v16 | ((unsigned __int8)v18 << 8);
LABEL_14:
    v13 = *(_BYTE *)a2;
LABEL_15:
    if ( v13 != 84 )
    {
      if ( *(_BYTE *)a4 != 84 )
        goto LABEL_17;
      v29 = v8;
      v21 = sub_D09500((__int64)a1, a4, a5, a2, (unsigned __int8 **)a3, (__int64)v8);
      v8 = v29;
      v22 = v21;
      v23 = (v21 >> 8) & 1;
      v24 = v22 >> 9;
      if ( (v22 & 0x100) != 0 )
      {
        if ( v24 == -4194304 )
        {
          v24 = -4194304;
        }
        else
        {
          LOBYTE(v23) = BYTE1(v22) & 1;
          v24 = (-512 * v24) >> 9;
        }
      }
      if ( (_BYTE)v22 != 1 )
        return (v24 << 9) | (unsigned __int8)v22 | ((unsigned __int8)v23 << 8);
LABEL_33:
      v13 = *(_BYTE *)a2;
LABEL_17:
      if ( v13 != 86 )
      {
        if ( *(_BYTE *)a4 != 86 )
          goto LABEL_19;
        v22 = sub_D04190((__int64)a1, a4, a5, (_BYTE *)a2, a3, v8);
        v23 = ((unsigned int)v22 >> 8) & 1;
        v24 = v22 >> 9;
        if ( (v22 & 0x100) != 0 )
        {
          if ( v24 == -4194304 )
          {
            v24 = -4194304;
          }
          else
          {
            LOBYTE(v23) = BYTE1(v22) & 1;
            v24 = (-512 * v24) >> 9;
          }
        }
        if ( (_BYTE)v22 == 1 )
          goto LABEL_19;
        return (v24 << 9) | (unsigned __int8)v22 | ((unsigned __int8)v23 << 8);
      }
LABEL_25:
      result = sub_D04190((__int64)a1, a2, a3, (_BYTE *)a4, a5, v8);
      if ( (_BYTE)result != 1 )
        return result;
      goto LABEL_19;
    }
LABEL_32:
    v30 = v8;
    result = sub_D09500((__int64)a1, a2, a3, a4, (unsigned __int8 **)a5, (__int64)v8);
    v8 = v30;
    if ( (_BYTE)result != 1 )
      return result;
    goto LABEL_33;
  }
  if ( v13 == 84 )
    goto LABEL_32;
  if ( v13 == 86 )
    goto LABEL_25;
LABEL_19:
  if ( a8 == a7
    && (v25 = sub_B2F070(a1[1], 0), a3 >= 0)
    && a5 >= 0
    && ((v26 = v25,
         (unsigned __int8)sub_D007E0(
                            (__int64)a8,
                            a3 & 0x3FFFFFFFFFFFFFFFLL,
                            (a3 & 0x4000000000000000LL) != 0,
                            *a1,
                            a1[2],
                            v25))
     || (unsigned __int8)sub_D007E0(
                           (__int64)a8,
                           a5 & 0x3FFFFFFFFFFFFFFFLL,
                           (a5 & 0x4000000000000000LL) != 0,
                           *a1,
                           a1[2],
                           v26)) )
  {
    return 2;
  }
  else
  {
    return 1;
  }
}
