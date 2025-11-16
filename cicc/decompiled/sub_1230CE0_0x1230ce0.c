// Function: sub_1230CE0
// Address: 0x1230ce0
//
__int64 __fastcall sub_1230CE0(__int64 a1, _QWORD *a2, __int64 *a3)
{
  __int16 v3; // r14
  int v5; // eax
  unsigned __int64 v6; // r15
  bool v7; // zf
  unsigned __int64 v8; // r10
  __int64 result; // rax
  int v10; // eax
  unsigned __int8 v11; // r15
  __int64 *v12; // rsi
  _QWORD *v13; // rax
  _QWORD *v14; // r12
  __int16 v15; // ax
  __int64 *v16; // r10
  int v17; // edx
  char v18; // al
  int v19; // eax
  char v20; // al
  char v21; // al
  unsigned __int64 v22; // [rsp+8h] [rbp-D8h]
  unsigned __int8 v24; // [rsp+1Ch] [rbp-C4h]
  unsigned int v25; // [rsp+1Ch] [rbp-C4h]
  char v26; // [rsp+21h] [rbp-BFh] BYREF
  __int16 v27; // [rsp+22h] [rbp-BEh] BYREF
  unsigned int v28; // [rsp+24h] [rbp-BCh] BYREF
  __int64 v29; // [rsp+28h] [rbp-B8h] BYREF
  unsigned __int64 v30; // [rsp+30h] [rbp-B0h] BYREF
  __int64 *v31; // [rsp+38h] [rbp-A8h] BYREF
  _QWORD v32[4]; // [rsp+40h] [rbp-A0h] BYREF
  __int16 v33; // [rsp+60h] [rbp-80h]
  const char *v34; // [rsp+70h] [rbp-70h] BYREF
  char *v35; // [rsp+78h] [rbp-68h]
  __int64 v36; // [rsp+80h] [rbp-60h]
  int v37; // [rsp+88h] [rbp-58h]
  char v38; // [rsp+8Ch] [rbp-54h]
  char v39; // [rsp+90h] [rbp-50h] BYREF
  char v40; // [rsp+91h] [rbp-4Fh]

  v3 = 0;
  v5 = *(_DWORD *)(a1 + 240);
  v29 = 0;
  v30 = 0;
  v27 = 0;
  v28 = 0;
  v31 = 0;
  if ( v5 == 248 )
  {
    v3 = 1;
    v5 = sub_1205200(a1 + 176);
    *(_DWORD *)(a1 + 240) = v5;
  }
  v24 = 0;
  if ( v5 == 239 )
  {
    v24 = 1;
    *(_DWORD *)(a1 + 240) = sub_1205200(a1 + 176);
  }
  v40 = 1;
  v6 = *(_QWORD *)(a1 + 232);
  v34 = "expected type";
  v39 = 3;
  if ( (unsigned __int8)sub_12190A0(a1, &v31, (int *)&v34, 0) )
    return 1;
  if ( *((_BYTE *)v31 + 8) == 13 || !sub_BCBD20((__int64)v31) )
  {
    v40 = 1;
    v39 = 3;
    v34 = "invalid type for alloca";
    sub_11FD800(a1 + 176, v6, (__int64)&v34, 1);
    return 1;
  }
  v7 = *(_DWORD *)(a1 + 240) == 4;
  v26 = 0;
  if ( !v7 )
  {
LABEL_9:
    v8 = 0;
    goto LABEL_10;
  }
  v10 = sub_1205200(a1 + 176);
  *(_DWORD *)(a1 + 240) = v10;
  switch ( v10 )
  {
    case 251:
      if ( !(unsigned __int8)sub_120CD10(a1, &v27, 0) && !(unsigned __int8)sub_1212970(a1, &v28, &v30, &v26) )
        goto LABEL_9;
      return 1;
    case 94:
      v30 = *(_QWORD *)(a1 + 232);
      if ( !(unsigned __int8)sub_1212650(a1, &v28, 0) )
        goto LABEL_9;
      return 1;
    case 511:
      v26 = 1;
      goto LABEL_9;
  }
  v22 = *(_QWORD *)(a1 + 232);
  if ( (unsigned __int8)sub_122FE20((__int64 **)a1, &v29, a3) )
    return 1;
  v8 = v22;
  if ( *(_DWORD *)(a1 + 240) != 4 )
    goto LABEL_10;
  v19 = sub_1205200(a1 + 176);
  v8 = v22;
  *(_DWORD *)(a1 + 240) = v19;
  switch ( v19 )
  {
    case 251:
      if ( !(unsigned __int8)sub_120CD10(a1, &v27, 0) )
      {
        v21 = sub_1212970(a1, &v28, &v30, &v26);
        v8 = v22;
        if ( !v21 )
          break;
      }
      return 1;
    case 94:
      v30 = *(_QWORD *)(a1 + 232);
      v20 = sub_1212650(a1, &v28, 0);
      v8 = v22;
      if ( !v20 )
        break;
      return 1;
    case 511:
      v26 = 1;
      break;
  }
LABEL_10:
  if ( v29 && *(_BYTE *)(*(_QWORD *)(v29 + 8) + 8LL) != 12 )
  {
    v40 = 1;
    v39 = 3;
    v34 = "element count must have integer type";
    sub_11FD800(a1 + 176, v8, (__int64)&v34, 1);
    return 1;
  }
  v34 = 0;
  v35 = &v39;
  v36 = 4;
  v37 = 0;
  v38 = 1;
  if ( HIBYTE(v27) )
    goto LABEL_21;
  v16 = v31;
  v17 = *((unsigned __int8 *)v31 + 8);
  if ( (_BYTE)v17 == 12 || (unsigned __int8)v17 <= 3u || (_BYTE)v17 == 5 || (v17 & 0xFB) == 0xA || (v17 & 0xFD) == 4 )
    goto LABEL_33;
  if ( ((unsigned __int8)(*((_BYTE *)v31 + 8) - 15) <= 3u || v17 == 20)
    && (unsigned __int8)sub_BCEBA0((__int64)v31, (__int64)&v34) )
  {
    if ( HIBYTE(v27) )
      goto LABEL_21;
    v16 = v31;
LABEL_33:
    v18 = sub_AE5260(*(_QWORD *)(a1 + 344) + 312LL, (__int64)v16);
    HIBYTE(v27) = 1;
    LOBYTE(v27) = v18;
LABEL_21:
    v11 = v27;
    v33 = 257;
    v12 = (__int64 *)unk_3F10A14;
    v13 = sub_BD2C40(80, unk_3F10A14);
    v14 = v13;
    if ( v13 )
    {
      v12 = v31;
      sub_B4CCA0((__int64)v13, v31, v28, v29, v11, (__int64)v32, 0, 0);
    }
    v15 = *((_WORD *)v14 + 1);
    LOBYTE(v15) = v15 & 0x3F;
    v7 = v26 == 0;
    *((_WORD *)v14 + 1) = v15 | (v3 << 6) | (v24 << 7);
    *a2 = v14;
    result = 2 * (unsigned int)!v7;
    goto LABEL_24;
  }
  v12 = (__int64 *)v6;
  v32[0] = "Cannot allocate unsized type";
  v33 = 259;
  sub_11FD800(a1 + 176, v6, (__int64)v32, 1);
  result = 1;
LABEL_24:
  if ( !v38 )
  {
    v25 = result;
    _libc_free(v35, v12);
    return v25;
  }
  return result;
}
