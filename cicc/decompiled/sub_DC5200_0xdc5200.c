// Function: sub_DC5200
// Address: 0xdc5200
//
_QWORD *__fastcall sub_DC5200(__int64 a1, __int64 a2, __int64 a3, unsigned int a4)
{
  __int64 *v4; // r15
  _QWORD *v5; // rsi
  __int64 **v6; // r13
  _QWORD *result; // rax
  __int64 v8; // r9
  __int64 v9; // rax
  unsigned int v10; // ebx
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // rsi
  __int64 v14; // r8
  __int64 v15; // r9
  __int64 *v16; // r8
  unsigned int v17; // ebx
  __int64 *v18; // r15
  __int64 v19; // rsi
  __int64 v20; // rax
  __int64 v21; // rdx
  __int64 v22; // rcx
  __int64 v23; // r8
  __int64 v24; // r9
  __int64 v25; // rdx
  __int64 v26; // rbx
  __int64 *v27; // rax
  __int64 v28; // r9
  __int16 v29; // ax
  int v30; // [rsp+38h] [rbp-128h]
  __int64 v32; // [rsp+48h] [rbp-118h]
  _QWORD *v33; // [rsp+50h] [rbp-110h]
  unsigned int v34; // [rsp+50h] [rbp-110h]
  _QWORD *v35; // [rsp+50h] [rbp-110h]
  __int64 *v36; // [rsp+50h] [rbp-110h]
  void *v37; // [rsp+50h] [rbp-110h]
  __int64 *v38; // [rsp+50h] [rbp-110h]
  __int64 v39; // [rsp+50h] [rbp-110h]
  unsigned int v40; // [rsp+50h] [rbp-110h]
  __int64 v41; // [rsp+58h] [rbp-108h] BYREF
  __int64 *v42; // [rsp+68h] [rbp-F8h] BYREF
  _BYTE *v43; // [rsp+70h] [rbp-F0h] BYREF
  __int64 v44; // [rsp+78h] [rbp-E8h]
  _BYTE v45[32]; // [rsp+80h] [rbp-E0h] BYREF
  _QWORD v46[2]; // [rsp+A0h] [rbp-C0h] BYREF
  int v47; // [rsp+B0h] [rbp-B0h] BYREF
  __int64 v48; // [rsp+B4h] [rbp-ACh]
  __int64 **v49; // [rsp+BCh] [rbp-A4h]

  v4 = (__int64 *)(a1 + 1032);
  v41 = a2;
  v5 = v46;
  v6 = (__int64 **)sub_D97090(a1, a3);
  v46[0] = &v47;
  v48 = v41;
  v49 = v6;
  v46[1] = 0x2000000005LL;
  v47 = 2;
  v42 = 0;
  result = sub_C65B40(a1 + 1032, (__int64)v46, (__int64 *)&v42, (__int64)off_49DEA80);
  if ( result )
    goto LABEL_2;
  v8 = v41;
  v9 = *(unsigned __int16 *)(v41 + 24);
  switch ( (_WORD)v9 )
  {
    case 0:
      v5 = (_QWORD *)sub_AD4C30(*(_QWORD *)(v41 + 32), v6, 0);
      result = sub_DA2570(a1, (__int64)v5);
      goto LABEL_2;
    case 2:
      v5 = *(_QWORD **)(v41 + 32);
      result = (_QWORD *)sub_DC5200(a1, v5, v6, a4 + 1);
      goto LABEL_2;
    case 4:
      v5 = *(_QWORD **)(v41 + 32);
      result = sub_DC5140(a1, (__int64)v5, (__int64)v6, a4 + 1);
      goto LABEL_2;
    case 3:
      v5 = *(_QWORD **)(v41 + 32);
      result = (_QWORD *)sub_DC5760(a1, v5, v6, a4 + 1);
      goto LABEL_2;
  }
  if ( a4 > dword_4F89148 )
    goto LABEL_34;
  if ( (unsigned __int16)(v9 - 5) <= 1u )
  {
    v43 = v45;
    v44 = 0x400000000LL;
    v30 = *(_QWORD *)(v41 + 40);
    if ( v30 )
    {
      v34 = 0;
      v10 = 0;
      do
      {
        v13 = sub_DC5200(a1, *(_QWORD *)(*(_QWORD *)(v41 + 32) + 8LL * v10), v6, a4 + 1);
        if ( (unsigned __int16)(*(_WORD *)(*(_QWORD *)(*(_QWORD *)(v41 + 32) + 8LL * v10) + 24LL) - 2) > 2u )
          v34 += *(_WORD *)(v13 + 24) == 2;
        ++v10;
        sub_D9B3A0((__int64)&v43, v13, v11, v12, v14, v15);
      }
      while ( v30 != v10 && v34 <= 1 );
      if ( v34 == 2 )
      {
        v5 = v46;
        result = sub_C65B40((__int64)v4, (__int64)v46, (__int64 *)&v42, (__int64)off_49DEA80);
        if ( result )
        {
LABEL_23:
          if ( v43 != v45 )
          {
            v35 = result;
            _libc_free(v43, v5);
            result = v35;
          }
          goto LABEL_2;
        }
        if ( v43 != v45 )
          _libc_free(v43, v46);
        v8 = v41;
        LOWORD(v9) = *(_WORD *)(v41 + 24);
        goto LABEL_28;
      }
      v8 = v41;
    }
    v29 = *(_WORD *)(v8 + 24);
    if ( v29 == 5 )
    {
      v5 = &v43;
      result = (_QWORD *)sub_DC7EB0(a1, &v43, 0, 0);
    }
    else
    {
      if ( v29 != 6 )
        BUG();
      v5 = &v43;
      result = (_QWORD *)sub_DC8BD0(a1, &v43, 0, 0);
    }
    goto LABEL_23;
  }
LABEL_28:
  if ( (_WORD)v9 == 8 )
  {
    v43 = v45;
    v44 = 0x400000000LL;
    v16 = *(__int64 **)(v8 + 32);
    v36 = &v16[*(_QWORD *)(v8 + 40)];
    if ( v36 != v16 )
    {
      v17 = a4;
      v32 = v8;
      v18 = *(__int64 **)(v8 + 32);
      do
      {
        v19 = *v18++;
        v20 = sub_DC5200(a1, v19, v6, v17 + 1);
        sub_D9B3A0((__int64)&v43, v20, v21, v22, v23, v24);
      }
      while ( v36 != v18 );
      v8 = v32;
    }
    v5 = &v43;
    result = sub_DBFF60(a1, (unsigned int *)&v43, *(_QWORD *)(v8 + 48), 0);
    goto LABEL_23;
  }
  v40 = sub_DB55F0(a1, v8);
  if ( v40 >= (unsigned __int64)sub_D97050(a1, (__int64)v6) )
  {
    v5 = v6;
    result = sub_DA2C50(a1, (__int64)v6, 0, 0);
    goto LABEL_2;
  }
LABEL_34:
  v37 = sub_C65D30((__int64)v46, (unsigned __int64 *)(a1 + 1064));
  v26 = v25;
  v27 = (__int64 *)sub_A777F0(0x30u, (__int64 *)(a1 + 1064));
  if ( v27 )
  {
    v28 = (__int64)v37;
    v38 = v27;
    sub_D96B70((__int64)v27, v28, v26, v41, (__int64)v6);
    v27 = v38;
  }
  v39 = (__int64)v27;
  sub_C657C0(v4, v27, v42, (__int64)off_49DEA80);
  v5 = (_QWORD *)v39;
  sub_DAEE00(a1, v39, &v41, 1);
  result = (_QWORD *)v39;
LABEL_2:
  if ( (int *)v46[0] != &v47 )
  {
    v33 = result;
    _libc_free(v46[0], v5);
    return v33;
  }
  return result;
}
