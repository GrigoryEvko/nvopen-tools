// Function: sub_1231270
// Address: 0x1231270
//
__int64 __fastcall sub_1231270(__int64 a1, _QWORD *a2, __int64 *a3)
{
  unsigned __int8 v3; // r14
  int v5; // eax
  char v6; // r15
  unsigned __int64 v7; // rax
  __int64 result; // rax
  __int64 *v9; // rdx
  int v10; // eax
  char v11; // r13
  __int64 *v12; // rsi
  _QWORD *v13; // rax
  _QWORD *v14; // r12
  const char *v15; // rax
  int v16; // ecx
  char v17; // al
  unsigned __int64 v18; // [rsp+8h] [rbp-D8h]
  unsigned __int64 v19; // [rsp+10h] [rbp-D0h]
  unsigned int v21; // [rsp+18h] [rbp-C8h]
  char v22; // [rsp+28h] [rbp-B8h] BYREF
  char v23; // [rsp+29h] [rbp-B7h] BYREF
  __int16 v24; // [rsp+2Ah] [rbp-B6h] BYREF
  int v25; // [rsp+2Ch] [rbp-B4h] BYREF
  __int64 v26; // [rsp+30h] [rbp-B0h] BYREF
  __int64 *v27; // [rsp+38h] [rbp-A8h] BYREF
  _QWORD v28[4]; // [rsp+40h] [rbp-A0h] BYREF
  __int16 v29; // [rsp+60h] [rbp-80h]
  const char *v30; // [rsp+70h] [rbp-70h] BYREF
  char *v31; // [rsp+78h] [rbp-68h]
  __int64 v32; // [rsp+80h] [rbp-60h]
  int v33; // [rsp+88h] [rbp-58h]
  char v34; // [rsp+8Ch] [rbp-54h]
  char v35; // [rsp+90h] [rbp-50h] BYREF
  char v36; // [rsp+91h] [rbp-4Fh]

  v3 = 0;
  v5 = *(_DWORD *)(a1 + 240);
  v24 = 0;
  v22 = 0;
  v25 = 0;
  v23 = 1;
  if ( v5 == 69 )
  {
    v3 = 1;
    v5 = sub_1205200(a1 + 176);
    *(_DWORD *)(a1 + 240) = v5;
  }
  v6 = 0;
  if ( v5 == 68 )
  {
    v6 = 1;
    *(_DWORD *)(a1 + 240) = sub_1205200(a1 + 176);
  }
  v7 = *(_QWORD *)(a1 + 232);
  v36 = 1;
  v19 = v7;
  v30 = "expected type";
  v35 = 3;
  if ( (unsigned __int8)sub_12190A0(a1, &v27, (int *)&v30, 0) )
    return 1;
  if ( (unsigned __int8)sub_120AFE0(a1, 4, "expected comma after load's type") )
    return 1;
  v18 = *(_QWORD *)(a1 + 232);
  if ( (unsigned __int8)sub_122FE20((__int64 **)a1, &v26, a3)
    || (unsigned __int8)sub_120E4B0(a1, v3, &v23, &v25)
    || (unsigned __int8)sub_120DF00(a1, &v24, &v22) )
  {
    return 1;
  }
  if ( *(_BYTE *)(*(_QWORD *)(v26 + 8) + 8LL) != 14
    || (v9 = v27, v10 = *((unsigned __int8 *)v27 + 8), v10 == 7)
    || v10 == 13 )
  {
    v36 = 1;
    v15 = "load operand must be a pointer to a first class type";
LABEL_25:
    v30 = v15;
    v35 = 3;
    sub_11FD800(a1 + 176, v18, (__int64)&v30, 1);
    return 1;
  }
  if ( v3 && !HIBYTE(v24) )
  {
    v36 = 1;
    v15 = "atomic load must have explicit non-zero alignment";
    goto LABEL_25;
  }
  if ( (unsigned int)(v25 - 5) <= 1 )
  {
    v36 = 1;
    v15 = "atomic load cannot use Release ordering";
    goto LABEL_25;
  }
  v30 = 0;
  v31 = &v35;
  v32 = 4;
  v33 = 0;
  v34 = 1;
  if ( HIBYTE(v24) )
  {
LABEL_19:
    v11 = v24;
    v29 = 257;
    v12 = (__int64 *)unk_3F10A14;
    v13 = sub_BD2C40(80, unk_3F10A14);
    v14 = v13;
    if ( v13 )
    {
      v12 = v27;
      sub_B4D0A0((__int64)v13, (__int64)v27, v26, (__int64)v28, v6 & 1, v11, v25, v23, 0, 0);
    }
    *a2 = v14;
    result = 2 * (unsigned int)(v22 != 0);
    goto LABEL_22;
  }
  v16 = *((unsigned __int8 *)v27 + 8);
  if ( (_BYTE)v16 == 12 || (unsigned __int8)v16 <= 3u || (_BYTE)v16 == 5 || (v16 & 0xFB) == 0xA || (v16 & 0xFD) == 4 )
  {
LABEL_32:
    v17 = sub_AE5020(*(_QWORD *)(a1 + 344) + 312LL, (__int64)v9);
    HIBYTE(v24) = 1;
    LOBYTE(v24) = v17;
    goto LABEL_19;
  }
  if ( ((unsigned __int8)(*((_BYTE *)v27 + 8) - 15) <= 3u || v16 == 20)
    && (unsigned __int8)sub_BCEBA0((__int64)v27, (__int64)&v30) )
  {
    if ( HIBYTE(v24) )
      goto LABEL_19;
    v9 = v27;
    goto LABEL_32;
  }
  v12 = (__int64 *)v19;
  v28[0] = "loading unsized types is not allowed";
  v29 = 259;
  sub_11FD800(a1 + 176, v19, (__int64)v28, 1);
  result = 1;
LABEL_22:
  if ( !v34 )
  {
    v21 = result;
    _libc_free(v31, v12);
    return v21;
  }
  return result;
}
