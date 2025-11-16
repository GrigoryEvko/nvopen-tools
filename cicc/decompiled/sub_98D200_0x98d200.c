// Function: sub_98D200
// Address: 0x98d200
//
_BOOL8 __fastcall sub_98D200(_BYTE *a1, __int64 a2, __int64 a3, __int64 a4, unsigned int a5)
{
  _BYTE *v5; // r14
  __int64 v6; // r13
  __int64 *v7; // rbx
  __int64 v8; // rax
  __int64 v9; // r13
  unsigned int v10; // r15d
  __int64 *v11; // r13
  __int64 v12; // rdx
  __int64 v13; // rcx
  unsigned int v14; // r8d
  __int64 v15; // rdx
  __int64 v16; // rcx
  unsigned int v17; // r8d
  __int64 v18; // rdx
  __int64 v19; // rcx
  unsigned int v20; // r8d
  __int64 v21; // rbx
  __int64 v23; // rax
  __int64 v24; // rdx
  __int64 v25; // r12
  int v26; // r12d
  __int64 v27; // rax
  __int64 v28; // rdx
  __int64 v29; // rsi
  __int64 v30; // rdx
  __int64 v31; // rcx
  unsigned int v32; // r8d
  __int64 v33; // rax
  int v34; // eax
  __int64 *v35; // [rsp+0h] [rbp-50h]
  int v36; // [rsp+Ch] [rbp-44h]
  __int64 *v37; // [rsp+10h] [rbp-40h]
  __int64 v38[7]; // [rsp+18h] [rbp-38h] BYREF

  v38[0] = (__int64)a1;
  v36 = a3;
  if ( a1 == (_BYTE *)a2 )
    return 1;
  if ( (unsigned int)a3 > 1 )
    return 0;
  v5 = (_BYTE *)a2;
  if ( *(_BYTE *)a2 <= 0x1Cu )
    return 0;
  v6 = 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
  {
    v7 = *(__int64 **)(a2 - 8);
    v35 = &v7[(unsigned __int64)v6 / 8];
  }
  else
  {
    v35 = (__int64 *)a2;
    v7 = (__int64 *)(a2 - v6);
  }
  v8 = v6 >> 5;
  v9 = v6 >> 7;
  if ( v9 )
  {
    v37 = &v7[16 * v9];
    v10 = a3 + 1;
    while ( 1 )
    {
      if ( (unsigned __int8)sub_98D0D0((__int64)v7, a2, a3, a4, a5) )
      {
        a2 = *v7;
        if ( (unsigned __int8)sub_98D200(a1, *v7, v10) )
          goto LABEL_14;
      }
      v11 = v7 + 4;
      if ( (unsigned __int8)sub_98D0D0((__int64)(v7 + 4), a2, v18, v19, v20) )
      {
        a2 = v7[4];
        if ( (unsigned __int8)sub_98D200(a1, a2, v10) )
        {
LABEL_29:
          v7 = v11;
          goto LABEL_14;
        }
      }
      v11 = v7 + 8;
      if ( (unsigned __int8)sub_98D0D0((__int64)(v7 + 8), a2, v12, v13, v14) )
      {
        a2 = v7[8];
        if ( (unsigned __int8)sub_98D200(a1, a2, v10) )
          goto LABEL_29;
        v11 = v7 + 12;
        if ( (unsigned __int8)sub_98D0D0((__int64)(v7 + 12), a2, v30, v31, v32) )
        {
LABEL_32:
          a2 = v7[12];
          if ( (unsigned __int8)sub_98D200(a1, a2, v10) )
            goto LABEL_29;
        }
      }
      else
      {
        v11 = v7 + 12;
        if ( (unsigned __int8)sub_98D0D0((__int64)(v7 + 12), a2, v15, v16, v17) )
          goto LABEL_32;
      }
      v7 += 16;
      if ( v7 == v37 )
      {
        v8 = ((char *)v35 - (char *)v7) >> 5;
        break;
      }
    }
  }
  if ( v8 == 2 )
  {
LABEL_43:
    if ( (unsigned __int8)sub_98D0D0((__int64)v7, a2, a3, a4, a5) )
    {
      a2 = *v7;
      if ( (unsigned __int8)sub_98D200(a1, *v7, (unsigned int)(v36 + 1)) )
        goto LABEL_14;
    }
    v7 += 4;
    goto LABEL_38;
  }
  if ( v8 == 3 )
  {
    if ( (unsigned __int8)sub_98D0D0((__int64)v7, a2, a3, a4, a5) )
    {
      a2 = *v7;
      if ( (unsigned __int8)sub_98D200(a1, *v7, (unsigned int)(v36 + 1)) )
        goto LABEL_14;
    }
    v7 += 4;
    goto LABEL_43;
  }
  if ( v8 != 1 )
    goto LABEL_15;
LABEL_38:
  if ( !(unsigned __int8)sub_98D0D0((__int64)v7, a2, a3, a4, a5)
    || !(unsigned __int8)sub_98D200(a1, *v7, (unsigned int)(v36 + 1)) )
  {
LABEL_15:
    if ( *v5 != 93 )
      return 0;
    v21 = *((_QWORD *)v5 - 4);
    if ( *(_BYTE *)v21 != 85 )
      return 0;
    v33 = *(_QWORD *)(v21 - 32);
    if ( !v33 || *(_BYTE *)v33 || *(_QWORD *)(v33 + 24) != *(_QWORD *)(v21 + 80) || (*(_BYTE *)(v33 + 33) & 0x20) == 0 )
      return 0;
    v34 = *(_DWORD *)(v33 + 36);
    if ( v34 != 312 )
    {
      switch ( v34 )
      {
        case 333:
        case 339:
        case 360:
        case 369:
        case 372:
          break;
        default:
          return 0;
      }
    }
    if ( *a1 == 93 && v21 == *((_QWORD *)a1 - 4) )
      return 1;
    if ( *(char *)(v21 + 7) < 0 )
    {
      v23 = sub_BD2BC0(*((_QWORD *)v5 - 4));
      v25 = v23 + v24;
      if ( *(char *)(v21 + 7) >= 0 )
      {
        if ( (unsigned int)(v25 >> 4) )
          goto LABEL_59;
      }
      else if ( (unsigned int)((v25 - sub_BD2BC0(v21)) >> 4) )
      {
        if ( *(char *)(v21 + 7) < 0 )
        {
          v26 = *(_DWORD *)(sub_BD2BC0(v21) + 8);
          if ( *(char *)(v21 + 7) >= 0 )
            BUG();
          v27 = sub_BD2BC0(v21);
          v29 = -32 - 32LL * (unsigned int)(*(_DWORD *)(v27 + v28 - 4) - v26);
          return (_QWORD *)(v21 + v29) != sub_9841D0(
                                            (_QWORD *)(v21 - 32LL * (*(_DWORD *)(v21 + 4) & 0x7FFFFFF)),
                                            v21 + v29,
                                            v38);
        }
LABEL_59:
        BUG();
      }
    }
    v29 = -32;
    return (_QWORD *)(v21 + v29) != sub_9841D0(
                                      (_QWORD *)(v21 - 32LL * (*(_DWORD *)(v21 + 4) & 0x7FFFFFF)),
                                      v21 + v29,
                                      v38);
  }
LABEL_14:
  if ( v7 == v35 )
    goto LABEL_15;
  return 1;
}
