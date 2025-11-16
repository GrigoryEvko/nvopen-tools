// Function: sub_34D5BE0
// Address: 0x34d5be0
//
unsigned __int64 __fastcall sub_34D5BE0(
        __int64 a1,
        int a2,
        __int64 a3,
        int *a4,
        unsigned __int64 a5,
        __int64 a6,
        signed int a7,
        __int64 a8)
{
  __int64 v9; // r13
  int v10; // edx
  int v13; // eax
  int v14; // r14d
  unsigned __int64 v15; // r15
  int v16; // r13d
  __int64 *v17; // rsi
  unsigned int v18; // eax
  unsigned __int64 v19; // r15
  __int64 *v20; // rsi
  unsigned int v21; // eax
  char v23; // al
  int v24; // r14d
  __int64 *v25; // rsi
  unsigned int v26; // eax
  unsigned __int64 v27; // r15
  __int64 *v28; // rsi
  unsigned int v29; // eax
  __int64 v30; // r8
  __int64 v31; // r9
  unsigned int v32; // eax
  int v33; // r13d
  unsigned __int64 v34; // r14
  int i; // r15d
  __int64 *v36; // rsi
  unsigned int v37; // eax
  int v38; // r14d
  __int64 *v39; // rsi
  unsigned int v40; // eax
  unsigned __int64 v41; // r15
  __int64 *v42; // rsi
  unsigned int v43; // eax
  int v44; // [rsp+Ch] [rbp-84h]
  int v45; // [rsp+Ch] [rbp-84h]
  int v46; // [rsp+Ch] [rbp-84h]
  int v47; // [rsp+Ch] [rbp-84h]
  int v48; // [rsp+Ch] [rbp-84h]
  char v49; // [rsp+1Fh] [rbp-71h] BYREF
  int v50; // [rsp+20h] [rbp-70h] BYREF
  signed int v51; // [rsp+24h] [rbp-6Ch] BYREF
  unsigned int **v52; // [rsp+28h] [rbp-68h] BYREF
  _QWORD v53[2]; // [rsp+30h] [rbp-60h] BYREF
  _QWORD v54[10]; // [rsp+40h] [rbp-50h] BYREF

  v9 = a8;
  if ( !a5 )
    goto LABEL_4;
  v10 = *(_DWORD *)(a3 + 32);
  if ( a2 != 6 )
  {
    if ( a2 != 7 )
    {
LABEL_4:
      switch ( a2 )
      {
        case 0:
          goto LABEL_39;
        case 1:
        case 2:
        case 3:
        case 6:
        case 7:
        case 8:
          goto LABEL_6;
        case 4:
          goto LABEL_24;
        case 5:
          goto LABEL_51;
        default:
          BUG();
      }
    }
    v44 = v10;
    if ( (unsigned __int8)sub_B4EDA0(a4, a5, v10) )
      goto LABEL_6;
    if ( (unsigned __int8)sub_B4EE20(a4, a5, v44) )
    {
LABEL_39:
      if ( *(_BYTE *)(a3 + 8) != 17 )
        return 0;
      v32 = sub_34D06B0(a1, **(__int64 ***)(a3 + 16));
      v33 = *(_DWORD *)(a3 + 32);
      v34 = v32;
      if ( v33 > 0 )
      {
        for ( i = 0; i != v33; ++i )
        {
          v36 = (__int64 *)a3;
          if ( (unsigned int)*(unsigned __int8 *)(a3 + 8) - 17 <= 1 )
            v36 = **(__int64 ***)(a3 + 16);
          v37 = sub_34D06B0(a1, v36);
          if ( __OFADD__(v37, v34) )
          {
            v34 = 0x8000000000000000LL;
            if ( v37 )
              v34 = 0x7FFFFFFFFFFFFFFFLL;
          }
          else
          {
            v34 += v37;
          }
        }
      }
      return v34;
    }
    v50 = v44;
    v54[1] = &v49;
    v54[2] = &v50;
    v54[0] = v53;
    v54[3] = &v51;
    v53[0] = a4;
    v53[1] = a5;
    v49 = 0;
    v51 = -1;
    v52 = (unsigned int **)v53;
    if ( sub_34D5B40(
           &v52,
           a5,
           (unsigned int)v44,
           (__int64)&v51,
           v30,
           v31,
           (__int64)v53,
           &v49,
           &v50,
           (unsigned int *)&v51) )
    {
      a7 = v51;
      goto LABEL_39;
    }
    if ( (unsigned __int8)sub_B4EFF0(a4, a5, v44, &a7) && a5 + a7 <= v44 )
    {
      v9 = sub_BCDA70(*(__int64 **)(a3 + 24), a5);
LABEL_51:
      v15 = 0;
      v47 = *(_DWORD *)(v9 + 32);
      if ( v47 )
      {
        v38 = 0;
        while ( 1 )
        {
          v39 = (__int64 *)a3;
          if ( (unsigned int)*(unsigned __int8 *)(a3 + 8) - 17 <= 1 )
            v39 = **(__int64 ***)(a3 + 16);
          v40 = sub_34D06B0(a1, v39);
          if ( !__OFADD__(v40, v15) )
            break;
          if ( v40 )
          {
            v41 = 0x7FFFFFFFFFFFFFFFLL;
LABEL_57:
            v42 = (__int64 *)v9;
            if ( (unsigned int)*(unsigned __int8 *)(v9 + 8) - 17 <= 1 )
LABEL_58:
              v42 = **(__int64 ***)(v9 + 16);
            v43 = sub_34D06B0(a1, v42);
            if ( __OFADD__(v43, v41) )
            {
              v15 = 0x8000000000000000LL;
              if ( v43 )
                v15 = 0x7FFFFFFFFFFFFFFFLL;
            }
            else
            {
              v15 = v43 + v41;
            }
            goto LABEL_61;
          }
          v41 = 0x8000000000000000LL;
          if ( (unsigned int)*(unsigned __int8 *)(v9 + 8) - 17 <= 1 )
            goto LABEL_58;
          v15 = (unsigned int)sub_34D06B0(a1, (__int64 *)v9) + 0x8000000000000000LL;
LABEL_61:
          if ( v47 == ++v38 )
            return v15;
        }
        v41 = v40 + v15;
        goto LABEL_57;
      }
      return v15;
    }
    goto LABEL_6;
  }
  if ( a5 <= 2 || (v45 = v10, v23 = sub_B4F0B0(a4, a5, v10, (int *)v54, &a7), v10 = v45, !v23) )
  {
    v48 = v10;
    if ( !(unsigned __int8)sub_B4EEA0(a4, a5, v10) && !(unsigned __int8)sub_B4EF10(a4, a5, v48) )
      sub_B4EF80((__int64)a4, a5, v48, &a7);
    goto LABEL_6;
  }
  if ( v45 >= LODWORD(v54[0]) + a7 )
  {
    v9 = sub_BCDA70(*(__int64 **)(a3 + 24), v54[0]);
LABEL_24:
    v15 = 0;
    v46 = *(_DWORD *)(v9 + 32);
    if ( !v46 )
      return v15;
    v24 = 0;
    while ( 1 )
    {
      v25 = (__int64 *)v9;
      if ( (unsigned int)*(unsigned __int8 *)(v9 + 8) - 17 <= 1 )
        v25 = **(__int64 ***)(v9 + 16);
      v26 = sub_34D06B0(a1, v25);
      if ( !__OFADD__(v26, v15) )
        break;
      if ( v26 )
      {
        v27 = 0x7FFFFFFFFFFFFFFFLL;
LABEL_30:
        v28 = (__int64 *)a3;
        if ( (unsigned int)*(unsigned __int8 *)(a3 + 8) - 17 <= 1 )
LABEL_31:
          v28 = **(__int64 ***)(a3 + 16);
        v29 = sub_34D06B0(a1, v28);
        if ( __OFADD__(v29, v27) )
        {
          v15 = 0x8000000000000000LL;
          if ( v29 )
            v15 = 0x7FFFFFFFFFFFFFFFLL;
        }
        else
        {
          v15 = v29 + v27;
        }
        goto LABEL_34;
      }
      v27 = 0x8000000000000000LL;
      if ( (unsigned int)*(unsigned __int8 *)(a3 + 8) - 17 <= 1 )
        goto LABEL_31;
      v15 = (unsigned int)sub_34D06B0(a1, (__int64 *)a3) + 0x8000000000000000LL;
LABEL_34:
      if ( v46 == ++v24 )
        return v15;
    }
    v27 = v26 + v15;
    goto LABEL_30;
  }
LABEL_6:
  v13 = *(unsigned __int8 *)(a3 + 8);
  if ( (_BYTE)v13 == 17 )
  {
    v14 = *(_DWORD *)(a3 + 32);
    v15 = 0;
    if ( v14 > 0 )
    {
      v16 = 0;
      while ( 1 )
      {
        v17 = (__int64 *)a3;
        if ( (unsigned int)(v13 - 17) <= 1 )
          v17 = **(__int64 ***)(a3 + 16);
        v18 = sub_34D06B0(a1, v17);
        if ( !__OFADD__(v18, v15) )
          break;
        if ( v18 )
        {
          v19 = 0x7FFFFFFFFFFFFFFFLL;
LABEL_14:
          v20 = (__int64 *)a3;
          if ( (unsigned int)*(unsigned __int8 *)(a3 + 8) - 17 > 1 )
            goto LABEL_16;
          goto LABEL_15;
        }
        v19 = 0x8000000000000000LL;
        if ( (unsigned int)*(unsigned __int8 *)(a3 + 8) - 17 > 1 )
        {
          v15 = (unsigned int)sub_34D06B0(a1, (__int64 *)a3) + 0x8000000000000000LL;
          goto LABEL_18;
        }
LABEL_15:
        v20 = **(__int64 ***)(a3 + 16);
LABEL_16:
        v21 = sub_34D06B0(a1, v20);
        if ( __OFADD__(v21, v19) )
        {
          v15 = 0x8000000000000000LL;
          if ( v21 )
            v15 = 0x7FFFFFFFFFFFFFFFLL;
        }
        else
        {
          v15 = v21 + v19;
        }
LABEL_18:
        if ( v14 == ++v16 )
          return v15;
        v13 = *(unsigned __int8 *)(a3 + 8);
      }
      v19 = v18 + v15;
      goto LABEL_14;
    }
    return v15;
  }
  return 0;
}
