// Function: sub_310C140
// Address: 0x310c140
//
_QWORD *__fastcall sub_310C140(__int64 *a1, __int64 a2)
{
  __int64 v4; // r8
  __int64 v5; // r9
  __int64 v6; // rdx
  __int64 v7; // rbx
  __int64 v8; // r11
  __int64 v9; // rsi
  __int64 v10; // r13
  __int64 v11; // rbx
  __int64 v12; // r13
  _QWORD *result; // rax
  __int64 v14; // rax
  signed __int64 v15; // r10
  __int64 v16; // r15
  __int64 v17; // rax
  _BYTE *v18; // r9
  unsigned int v19; // r10d
  __int64 v20; // r8
  _QWORD *v21; // rax
  _BYTE *v22; // rdi
  __int64 v23; // rbx
  _QWORD *v24; // rax
  __int64 v25; // rdx
  __int64 v26; // rsi
  _BYTE *v27; // rdi
  _BYTE *src; // [rsp+8h] [rbp-D8h]
  size_t v29; // [rsp+10h] [rbp-D0h]
  signed __int64 n; // [rsp+18h] [rbp-C8h]
  size_t na; // [rsp+18h] [rbp-C8h]
  unsigned int nb; // [rsp+18h] [rbp-C8h]
  __int64 v33; // [rsp+20h] [rbp-C0h]
  unsigned int v34; // [rsp+20h] [rbp-C0h]
  __int64 v35; // [rsp+20h] [rbp-C0h]
  __int64 v36; // [rsp+28h] [rbp-B8h]
  _QWORD *v37; // [rsp+30h] [rbp-B0h]
  __int64 v38; // [rsp+30h] [rbp-B0h]
  __int64 v39; // [rsp+30h] [rbp-B0h]
  __int64 v40; // [rsp+40h] [rbp-A0h] BYREF
  __int64 v41; // [rsp+48h] [rbp-98h] BYREF
  _BYTE *v42; // [rsp+50h] [rbp-90h] BYREF
  __int64 v43; // [rsp+58h] [rbp-88h]
  _BYTE v44[32]; // [rsp+60h] [rbp-80h] BYREF
  _BYTE *v45; // [rsp+80h] [rbp-60h] BYREF
  __int64 v46; // [rsp+88h] [rbp-58h]
  _BYTE v47[80]; // [rsp+90h] [rbp-50h] BYREF

  if ( *(_QWORD *)(a2 + 40) != 2 )
    return (_QWORD *)sub_310A840(a1, a2);
  sub_310BF50(*a1, **(_QWORD **)(a2 + 32), a1[1], &v40, &v41);
  v6 = *(_QWORD *)(a2 + 40);
  v7 = a1[1];
  v8 = *a1;
  if ( v6 == 2 )
  {
    v9 = *(_QWORD *)(*(_QWORD *)(a2 + 32) + 8LL);
    goto LABEL_4;
  }
  v14 = *(_QWORD *)(a2 + 48);
  v15 = 8 * v6 - 8;
  v42 = v44;
  v16 = v15 >> 3;
  v36 = v14;
  v17 = *(_QWORD *)(a2 + 32);
  v43 = 0x300000000LL;
  if ( (unsigned __int64)v15 > 0x18 )
  {
    n = 8 * v6 - 8;
    v33 = v17;
    v38 = v8;
    sub_C8D5F0((__int64)&v42, v44, n >> 3, 8u, v4, v5);
    v8 = v38;
    v17 = v33;
    v15 = n;
    v22 = &v42[8 * (unsigned int)v43];
  }
  else
  {
    v18 = v44;
    if ( 8 * v6 == 8 )
      goto LABEL_9;
    v22 = v44;
  }
  v39 = v8;
  memcpy(v22, (const void *)(v17 + 8), v15);
  v18 = v42;
  LODWORD(v15) = v43;
  v8 = v39;
LABEL_9:
  LODWORD(v43) = v16 + v15;
  v19 = v16 + v15;
  v20 = 8LL * v19;
  v45 = v47;
  v46 = 0x400000000LL;
  if ( v19 > 4uLL )
  {
    src = v18;
    v29 = 8LL * v19;
    na = v8;
    v34 = v19;
    sub_C8D5F0((__int64)&v45, v47, v19, 8u, v20, (__int64)v18);
    v19 = v34;
    v8 = na;
    v20 = v29;
    v18 = src;
    v27 = &v45[8 * (unsigned int)v46];
LABEL_22:
    nb = v19;
    v35 = v8;
    memcpy(v27, v18, v20);
    LODWORD(v20) = v46;
    v19 = nb;
    v8 = v35;
    goto LABEL_11;
  }
  if ( v20 )
  {
    v27 = v47;
    goto LABEL_22;
  }
LABEL_11:
  LODWORD(v46) = v19 + v20;
  v21 = sub_DBFF60(v8, (unsigned int *)&v45, v36, 0);
  v9 = (__int64)v21;
  if ( v45 != v47 )
  {
    v37 = v21;
    _libc_free((unsigned __int64)v45);
    v9 = (__int64)v37;
  }
  if ( v42 != v44 )
    _libc_free((unsigned __int64)v42);
  v8 = *a1;
LABEL_4:
  sub_310BF50(v8, v9, v7, (__int64 *)&v42, (__int64 *)&v45);
  v10 = sub_D95540(a1[1]);
  v11 = sub_D95540(v40);
  if ( v11 != v10 )
    return (_QWORD *)sub_310A840(a1, a2);
  v12 = sub_D95540(v41);
  if ( v12 != v11 )
    return (_QWORD *)sub_310A840(a1, a2);
  v23 = sub_D95540((__int64)v42);
  if ( v23 != v12 || v23 != sub_D95540((__int64)v45) )
    return (_QWORD *)sub_310A840(a1, a2);
  v24 = sub_DC1960(*a1, v40, (__int64)v42, *(_QWORD *)(a2 + 48), *(_WORD *)(a2 + 28) & 7);
  v25 = (__int64)v45;
  v26 = v41;
  a1[2] = (__int64)v24;
  result = sub_DC1960(*a1, v26, v25, *(_QWORD *)(a2 + 48), *(_WORD *)(a2 + 28) & 7);
  a1[3] = (__int64)result;
  return result;
}
