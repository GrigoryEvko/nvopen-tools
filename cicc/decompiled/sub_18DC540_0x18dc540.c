// Function: sub_18DC540
// Address: 0x18dc540
//
__int64 __fastcall sub_18DC540(__int64 a1, __int64 a2, __int64 *a3, int a4)
{
  __int64 v5; // rax
  unsigned __int64 v6; // r14
  unsigned __int8 v7; // al
  _BYTE *v8; // r15
  unsigned __int8 v9; // al
  __int64 v10; // r13
  __int64 result; // rax
  unsigned __int64 v12; // r8
  __int64 i; // rdi
  int v14; // edi
  __int64 v15; // rax
  __int64 v16; // r15
  unsigned __int8 v17; // al
  int v18; // eax
  unsigned __int8 v19; // al
  __int64 v20; // rbx
  unsigned __int64 v21; // r8
  __int64 v22; // rax
  __int64 *v23; // rbx
  _BYTE *v24; // r13
  unsigned __int8 v25; // al
  __int64 v26; // r15
  _BYTE *v27; // rax
  _BYTE *v28; // rax
  __int64 *v29; // rbx
  _BYTE *v30; // r13
  unsigned __int8 v31; // al
  __int64 v32; // r15
  _BYTE *v33; // rax
  _BYTE *v34; // rax
  __int64 *v35; // [rsp+18h] [rbp-78h]
  unsigned __int64 v36; // [rsp+18h] [rbp-78h]
  unsigned __int64 v37; // [rsp+28h] [rbp-68h] BYREF
  _BYTE *v38; // [rsp+30h] [rbp-60h] BYREF
  __int64 v39; // [rsp+38h] [rbp-58h]
  __int64 v40; // [rsp+40h] [rbp-50h]
  __int64 v41; // [rsp+48h] [rbp-48h]
  __int64 v42; // [rsp+50h] [rbp-40h]

  if ( a4 == 22 )
    return 0;
  v5 = sub_15F2050(a1);
  v6 = sub_1632FA0(v5);
  v7 = *(_BYTE *)(a1 + 16);
  if ( v7 == 75 )
  {
    v8 = *(_BYTE **)(a1 - 24);
    v9 = v8[16];
    if ( v9 <= 0x10u )
      return 0;
    if ( v9 == 53 )
      return 0;
    v10 = *a3;
    if ( v9 == 17
      && ((unsigned __int8)sub_15E0450(*(_QWORD *)(a1 - 24))
       || (unsigned __int8)sub_15E0470((__int64)v8)
       || (unsigned __int8)sub_15E0490((__int64)v8)
       || (unsigned __int8)sub_15E04F0((__int64)v8)) )
    {
      return 0;
    }
    if ( *(_BYTE *)(*(_QWORD *)v8 + 8LL) != 15 )
      return 0;
    v38 = v8;
    v39 = -1;
    v40 = 0;
    v41 = 0;
    v42 = 0;
    if ( (unsigned __int8)sub_134CBB0(v10, (__int64)&v38, 0) )
      return 0;
    if ( v8[16] == 54 )
    {
      v28 = (_BYTE *)*((_QWORD *)v8 - 3);
      v39 = -1;
      v40 = 0;
      v38 = v28;
      v41 = 0;
      v42 = 0;
      if ( (unsigned __int8)sub_134CBB0(v10, (__int64)&v38, 0) )
        return 0;
    }
    goto LABEL_34;
  }
  v37 = 0;
  if ( v7 > 0x17u )
  {
    v12 = a1 | 4;
    if ( v7 == 78 || (v12 = a1 & 0xFFFFFFFFFFFFFFFBLL, v7 == 29) )
    {
      v37 = v12;
      v21 = v12 & 0xFFFFFFFFFFFFFFF8LL;
      if ( v21 )
      {
        v29 = (__int64 *)(v21 - 24LL * (*(_DWORD *)(v21 + 20) & 0xFFFFFFF));
        v36 = sub_134EF80(&v37);
        if ( (__int64 *)v36 == v29 )
          return 0;
        while ( 1 )
        {
          while ( 1 )
          {
            v30 = (_BYTE *)*v29;
            v31 = *(_BYTE *)(*v29 + 16);
            if ( v31 > 0x10u && v31 != 53 )
            {
              v32 = *a3;
              if ( (v31 != 17
                 || !(unsigned __int8)sub_15E0450(*v29)
                 && !(unsigned __int8)sub_15E0470((__int64)v30)
                 && !(unsigned __int8)sub_15E0490((__int64)v30)
                 && !(unsigned __int8)sub_15E04F0((__int64)v30))
                && *(_BYTE *)(*(_QWORD *)v30 + 8LL) == 15 )
              {
                v38 = v30;
                v39 = -1;
                v40 = 0;
                v41 = 0;
                v42 = 0;
                if ( !(unsigned __int8)sub_134CBB0(v32, (__int64)&v38, 0) )
                {
                  if ( v30[16] != 54 )
                    break;
                  v33 = (_BYTE *)*((_QWORD *)v30 - 3);
                  v39 = -1;
                  v38 = v33;
                  v40 = 0;
                  v41 = 0;
                  v42 = 0;
                  if ( !(unsigned __int8)sub_134CBB0(v32, (__int64)&v38, 0) )
                    break;
                }
              }
            }
            v29 += 3;
            if ( (__int64 *)v36 == v29 )
              return 0;
          }
          result = sub_18DDD00(a3, a2, v30, v6);
          if ( (_BYTE)result )
            break;
          v29 += 3;
          if ( (__int64 *)v36 == v29 )
            return 0;
        }
        return result;
      }
      goto LABEL_34;
    }
  }
  if ( v7 != 55 )
  {
LABEL_34:
    v22 = 3LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF);
    if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
    {
      v23 = *(__int64 **)(a1 - 8);
      v35 = &v23[v22];
    }
    else
    {
      v35 = (__int64 *)a1;
      v23 = (__int64 *)(a1 - v22 * 8);
    }
    if ( v35 == v23 )
      return 0;
    while ( 1 )
    {
      v24 = (_BYTE *)*v23;
      v25 = *(_BYTE *)(*v23 + 16);
      if ( v25 > 0x10u && v25 != 53 )
      {
        v26 = *a3;
        if ( (v25 != 17
           || !(unsigned __int8)sub_15E0450(*v23)
           && !(unsigned __int8)sub_15E0470((__int64)v24)
           && !(unsigned __int8)sub_15E0490((__int64)v24)
           && !(unsigned __int8)sub_15E04F0((__int64)v24))
          && *(_BYTE *)(*(_QWORD *)v24 + 8LL) == 15 )
        {
          v38 = v24;
          v39 = -1;
          v40 = 0;
          v41 = 0;
          v42 = 0;
          if ( !(unsigned __int8)sub_134CBB0(v26, (__int64)&v38, 0) )
          {
            if ( v24[16] != 54
              || (v27 = (_BYTE *)*((_QWORD *)v24 - 3),
                  v39 = -1,
                  v38 = v27,
                  v40 = 0,
                  v41 = 0,
                  v42 = 0,
                  !(unsigned __int8)sub_134CBB0(v26, (__int64)&v38, 0)) )
            {
              result = sub_18DDD00(a3, a2, v24, v6);
              if ( (_BYTE)result )
                break;
            }
          }
        }
      }
      v23 += 3;
      if ( v23 == v35 )
        return 0;
    }
    return result;
  }
  for ( i = *(_QWORD *)(a1 - 24); ; i = *(_QWORD *)(v16 - 24LL * (*(_DWORD *)(v16 + 20) & 0xFFFFFFF)) )
  {
    v15 = sub_14AD280(i, v6, 6u);
    v14 = 23;
    v16 = v15;
    v17 = *(_BYTE *)(v15 + 16);
    if ( v17 > 0x17u )
    {
      if ( v17 != 78 )
      {
        v14 = 2 * (v17 != 29) + 21;
        goto LABEL_16;
      }
      v14 = 21;
      if ( !*(_BYTE *)(*(_QWORD *)(v16 - 24) + 16LL) )
        break;
    }
LABEL_16:
    if ( !(unsigned __int8)sub_1439C90(v14) )
      goto LABEL_22;
LABEL_17:
    ;
  }
  v18 = sub_1438F00(*(_QWORD *)(v16 - 24));
  if ( (unsigned __int8)sub_1439C90(v18) )
    goto LABEL_17;
LABEL_22:
  v19 = *(_BYTE *)(v16 + 16);
  if ( v19 <= 0x10u )
    return 0;
  if ( v19 == 53 )
    return 0;
  v20 = *a3;
  if ( v19 == 17
    && ((unsigned __int8)sub_15E0450(v16)
     || (unsigned __int8)sub_15E0470(v16)
     || (unsigned __int8)sub_15E0490(v16)
     || (unsigned __int8)sub_15E04F0(v16)) )
  {
    return 0;
  }
  if ( *(_BYTE *)(*(_QWORD *)v16 + 8LL) != 15 )
    return 0;
  v38 = (_BYTE *)v16;
  v39 = -1;
  v40 = 0;
  v41 = 0;
  v42 = 0;
  if ( (unsigned __int8)sub_134CBB0(v20, (__int64)&v38, 0) )
    return 0;
  if ( *(_BYTE *)(v16 + 16) == 54
    && (v34 = *(_BYTE **)(v16 - 24),
        v39 = -1,
        v40 = 0,
        v38 = v34,
        v41 = 0,
        v42 = 0,
        (unsigned __int8)sub_134CBB0(v20, (__int64)&v38, 0)) )
  {
    return 0;
  }
  else
  {
    return sub_18DDD00(a3, v16, a2, v6);
  }
}
