// Function: sub_1491390
// Address: 0x1491390
//
__int64 __fastcall sub_1491390(__int64 a1, __int64 a2, __int64 a3, unsigned int a4, __m128i a5, __m128i a6)
{
  _QWORD *v7; // r12
  __int64 v8; // r15
  __int64 v9; // rax
  __int64 v10; // r9
  __int64 *v11; // rbx
  __int64 v12; // rax
  __int64 *v13; // r8
  unsigned int v14; // eax
  __int64 v15; // rdx
  _QWORD *v16; // r13
  __int64 v17; // r12
  __int64 *v18; // rax
  __int64 v19; // r9
  __int64 v20; // rax
  __int64 v21; // r9
  __int64 v22; // rbx
  __int64 v23; // rax
  int v24; // ebx
  __int64 v25; // rax
  __int64 v26; // rbx
  __int64 v27; // rax
  __int64 v28; // rbx
  __int64 v29; // rax
  __int64 v30; // r9
  __int64 v31; // rax
  __int64 v33; // rbx
  __int64 v34; // rax
  __int64 *v35; // r12
  __int64 v36; // rax
  int v37; // ebx
  __int64 v38; // rax
  __int64 v39; // rbx
  __int64 v40; // rax
  __int64 v41; // rbx
  bool v42; // al
  char v43; // al
  __int64 v44; // [rsp+0h] [rbp-B0h]
  __int64 v45; // [rsp+8h] [rbp-A8h]
  unsigned int v46; // [rsp+8h] [rbp-A8h]
  __int64 v47; // [rsp+8h] [rbp-A8h]
  __int64 v48; // [rsp+10h] [rbp-A0h]
  __int64 v49; // [rsp+10h] [rbp-A0h]
  __int64 *v50; // [rsp+10h] [rbp-A0h]
  __int64 v51; // [rsp+10h] [rbp-A0h]
  __int64 v52; // [rsp+18h] [rbp-98h]
  __int64 v53; // [rsp+20h] [rbp-90h]
  __int64 v54; // [rsp+28h] [rbp-88h]
  __int64 v55; // [rsp+28h] [rbp-88h]
  unsigned int v56; // [rsp+4Ch] [rbp-64h] BYREF
  _QWORD *v57; // [rsp+50h] [rbp-60h] BYREF
  __int64 v58; // [rsp+58h] [rbp-58h]
  _QWORD v59[10]; // [rsp+60h] [rbp-50h] BYREF

  v7 = (_QWORD *)a3;
  v53 = *(_QWORD *)(a1 + 48);
  v8 = **(_QWORD **)(a1 + 32);
  v9 = sub_13A5BC0((_QWORD *)a1, a3);
  if ( *(_WORD *)(v8 + 24) != 4 )
    return sub_14747F0((__int64)v7, **(_QWORD **)(a1 + 32), a2, a4);
  v10 = v9;
  v57 = v59;
  v58 = 0x400000000LL;
  v11 = *(__int64 **)(v8 + 32);
  v12 = *(_QWORD *)(v8 + 40);
  v13 = &v11[v12];
  if ( v11 == v13 )
  {
    if ( !v12 )
      return sub_14747F0((__int64)v7, **(_QWORD **)(a1 + 32), a2, a4);
  }
  else
  {
    v14 = a4;
    v15 = 0;
    v16 = v7;
    do
    {
      v17 = *v11;
      if ( v10 != *v11 )
      {
        if ( HIDWORD(v58) <= (unsigned int)v15 )
        {
          v46 = v14;
          v50 = v13;
          v52 = v10;
          sub_16CD150(&v57, v59, 0, 8);
          v15 = (unsigned int)v58;
          v14 = v46;
          v13 = v50;
          v10 = v52;
        }
        v57[v15] = v17;
        v15 = (unsigned int)(v58 + 1);
        LODWORD(v58) = v58 + 1;
      }
      ++v11;
    }
    while ( v13 != v11 );
    v7 = v16;
    a4 = v14;
    if ( v15 == *(_QWORD *)(v8 + 40) )
    {
LABEL_15:
      if ( v57 != v59 )
        _libc_free((unsigned __int64)v57);
      return sub_14747F0((__int64)v7, **(_QWORD **)(a1 + 32), a2, a4);
    }
  }
  v54 = v10;
  v18 = sub_147DD40((__int64)v7, (__int64 *)&v57, *(_WORD *)(v8 + 26) & 2, 0, a5, a6);
  v19 = v54;
  v55 = (__int64)v18;
  v48 = v19;
  v45 = sub_14799E0((__int64)v7, (__int64)v18, v19, v53, 0);
  if ( *(_WORD *)(v45 + 24) != 7 )
  {
    v47 = v48;
    sub_1481F60(v7, v53, a5, a6);
    v36 = sub_1456040(**(_QWORD **)(a1 + 32));
    v37 = sub_1456C90((__int64)v7, v36);
    v38 = sub_15E0530(v7[3]);
    v51 = sub_1644900(v38, (unsigned int)(2 * v37));
    v39 = sub_14747F0((__int64)v7, v47, v51, a4);
    v40 = sub_14747F0((__int64)v7, v55, v51, a4);
    v41 = sub_13A5B00((__int64)v7, v40, v39, 0, 0);
    if ( v41 == sub_14747F0((__int64)v7, v8, v51, a4) )
      goto LABEL_22;
    v30 = v47;
LABEL_13:
    v31 = sub_1477D10(v30, &v56, (__int64)v7);
    if ( v31 && (unsigned __int8)sub_148B410((__int64)v7, v53, v56, v55, v31) )
      goto LABEL_22;
    goto LABEL_15;
  }
  v20 = sub_1481F60(v7, v53, a5, a6);
  v21 = v48;
  v22 = v20;
  if ( (*(_BYTE *)(v45 + 26) & 2) != 0 )
  {
    v42 = sub_14562D0(v20);
    v21 = v48;
    if ( !v42 )
    {
      v43 = sub_1477C30((__int64)v7, v22);
      v21 = v48;
      if ( v43 )
        goto LABEL_22;
    }
  }
  v44 = v21;
  v23 = sub_1456040(**(_QWORD **)(a1 + 32));
  v24 = sub_1456C90((__int64)v7, v23);
  v25 = sub_15E0530(v7[3]);
  v49 = sub_1644900(v25, (unsigned int)(2 * v24));
  v26 = sub_14747F0((__int64)v7, v44, v49, a4);
  v27 = sub_14747F0((__int64)v7, v55, v49, a4);
  v28 = sub_13A5B00((__int64)v7, v27, v26, 0, 0);
  v29 = sub_14747F0((__int64)v7, v8, v49, a4);
  v30 = v44;
  if ( v28 != v29 )
    goto LABEL_13;
  if ( (*(_BYTE *)(a1 + 26) & 2) != 0 )
    *(_WORD *)(v45 + 26) |= 3u;
LABEL_22:
  if ( v57 != v59 )
    _libc_free((unsigned __int64)v57);
  if ( !v55 )
    return sub_14747F0((__int64)v7, **(_QWORD **)(a1 + 32), a2, a4);
  v33 = sub_14747F0((__int64)v7, v55, a2, a4);
  v34 = sub_13A5BC0((_QWORD *)a1, (__int64)v7);
  v59[0] = sub_14747F0((__int64)v7, v34, a2, a4);
  v57 = v59;
  v59[1] = v33;
  v58 = 0x200000002LL;
  v35 = sub_147DD40((__int64)v7, (__int64 *)&v57, 0, 0, a5, a6);
  if ( v57 != v59 )
    _libc_free((unsigned __int64)v57);
  return (__int64)v35;
}
