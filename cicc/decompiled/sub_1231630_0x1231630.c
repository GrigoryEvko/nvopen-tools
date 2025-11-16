// Function: sub_1231630
// Address: 0x1231630
//
__int64 __fastcall sub_1231630(__int64 a1, _QWORD *a2, __int64 *a3)
{
  unsigned __int8 v3; // r13
  int v6; // eax
  char v7; // r14
  unsigned __int64 v8; // r15
  __int64 result; // rax
  __int64 v10; // rdx
  int v11; // eax
  char v12; // r13
  __int64 v13; // rsi
  _QWORD *v14; // r12
  const char *v15; // rax
  __int64 v16; // rdi
  int v17; // ecx
  char v18; // al
  unsigned __int64 v19; // [rsp+0h] [rbp-D0h]
  unsigned int v21; // [rsp+8h] [rbp-C8h]
  char v22; // [rsp+18h] [rbp-B8h] BYREF
  char v23; // [rsp+19h] [rbp-B7h] BYREF
  __int16 v24; // [rsp+1Ah] [rbp-B6h] BYREF
  int v25; // [rsp+1Ch] [rbp-B4h] BYREF
  __int64 v26; // [rsp+20h] [rbp-B0h] BYREF
  __int64 v27; // [rsp+28h] [rbp-A8h] BYREF
  const char *v28; // [rsp+30h] [rbp-A0h] BYREF
  char v29; // [rsp+50h] [rbp-80h]
  char v30; // [rsp+51h] [rbp-7Fh]
  const char *v31; // [rsp+60h] [rbp-70h] BYREF
  char *v32; // [rsp+68h] [rbp-68h]
  __int64 v33; // [rsp+70h] [rbp-60h]
  int v34; // [rsp+78h] [rbp-58h]
  char v35; // [rsp+7Ch] [rbp-54h]
  char v36; // [rsp+80h] [rbp-50h] BYREF
  char v37; // [rsp+81h] [rbp-4Fh]

  v3 = 0;
  v24 = 0;
  v6 = *(_DWORD *)(a1 + 240);
  v22 = 0;
  v25 = 0;
  v23 = 1;
  if ( v6 == 69 )
  {
    v3 = 1;
    v6 = sub_1205200(a1 + 176);
    *(_DWORD *)(a1 + 240) = v6;
  }
  v7 = 0;
  if ( v6 == 68 )
  {
    v7 = 1;
    *(_DWORD *)(a1 + 240) = sub_1205200(a1 + 176);
  }
  v8 = *(_QWORD *)(a1 + 232);
  if ( (unsigned __int8)sub_122FE20((__int64 **)a1, &v26, a3) )
    return 1;
  if ( (unsigned __int8)sub_120AFE0(a1, 4, "expected ',' after store operand") )
    return 1;
  v19 = *(_QWORD *)(a1 + 232);
  if ( (unsigned __int8)sub_122FE20((__int64 **)a1, &v27, a3)
    || (unsigned __int8)sub_120E4B0(a1, v3, &v23, &v25)
    || (unsigned __int8)sub_120DF00(a1, &v24, &v22) )
  {
    return 1;
  }
  if ( *(_BYTE *)(*(_QWORD *)(v27 + 8) + 8LL) != 14 )
  {
    v37 = 1;
    v31 = "store operand must be a pointer";
    v36 = 3;
    sub_11FD800(a1 + 176, v19, (__int64)&v31, 1);
    return 1;
  }
  v10 = v26;
  v11 = *(unsigned __int8 *)(*(_QWORD *)(v26 + 8) + 8LL);
  if ( v11 == 7 || v11 == 13 )
  {
    v37 = 1;
    v15 = "store operand must be a first class value";
LABEL_26:
    v31 = v15;
    v36 = 3;
    sub_11FD800(a1 + 176, v8, (__int64)&v31, 1);
    return 1;
  }
  if ( v3 && !HIBYTE(v24) )
  {
    v37 = 1;
    v15 = "atomic store must have explicit non-zero alignment";
    goto LABEL_26;
  }
  if ( (v25 & 0xFFFFFFFD) == 4 )
  {
    v37 = 1;
    v15 = "atomic store cannot use Acquire ordering";
    goto LABEL_26;
  }
  v31 = 0;
  v32 = &v36;
  v33 = 4;
  v34 = 0;
  v35 = 1;
  if ( HIBYTE(v24) )
  {
LABEL_20:
    v12 = v24;
    v13 = unk_3F10A10;
    v14 = sub_BD2C40(80, unk_3F10A10);
    if ( v14 )
    {
      v13 = v26;
      sub_B4D260((__int64)v14, v26, v27, v7, v12, v25, v23, 0, 0);
    }
    *a2 = v14;
    result = 2 * (unsigned int)(v22 != 0);
    goto LABEL_23;
  }
  v16 = *(_QWORD *)(v26 + 8);
  v17 = *(unsigned __int8 *)(v16 + 8);
  if ( (_BYTE)v17 == 12 || (unsigned __int8)v17 <= 3u || (_BYTE)v17 == 5 || (v17 & 0xFB) == 0xA || (v17 & 0xFD) == 4 )
  {
LABEL_33:
    v18 = sub_AE5020(*(_QWORD *)(a1 + 344) + 312LL, *(_QWORD *)(v10 + 8));
    HIBYTE(v24) = 1;
    LOBYTE(v24) = v18;
    goto LABEL_20;
  }
  if ( ((unsigned __int8)(*(_BYTE *)(v16 + 8) - 15) <= 3u || v17 == 20)
    && (unsigned __int8)sub_BCEBA0(v16, (__int64)&v31) )
  {
    if ( HIBYTE(v24) )
      goto LABEL_20;
    v10 = v26;
    goto LABEL_33;
  }
  v13 = v8;
  v28 = "storing unsized types is not allowed";
  v30 = 1;
  v29 = 3;
  sub_11FD800(a1 + 176, v8, (__int64)&v28, 1);
  result = 1;
LABEL_23:
  if ( !v35 )
  {
    v21 = result;
    _libc_free(v32, v13);
    return v21;
  }
  return result;
}
