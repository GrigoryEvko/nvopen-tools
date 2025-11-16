// Function: sub_F560C0
// Address: 0xf560c0
//
__int64 __fastcall sub_F560C0(unsigned __int8 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  int v7; // edx
  __int64 v8; // r14
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rbx
  __int64 v12; // rdx
  int v13; // ebx
  __int64 v14; // rax
  __int64 v15; // rdx
  int v16; // edx
  unsigned __int8 *v17; // r13
  int v18; // ecx
  __int64 *v19; // rax
  __int64 v20; // rdx
  unsigned __int8 *v21; // r12
  __int64 v22; // rdx
  __int64 v23; // r14
  _BYTE *v24; // r14
  _BYTE *v25; // rdi
  _BYTE *v26; // rdx
  int v27; // esi
  __int64 v28; // rax
  __int64 v29; // rsi
  _QWORD *v30; // rax
  __int64 v31; // r12
  __int64 v32; // r14
  __int64 *v33; // r8
  __int64 v34; // rsi
  __int64 v35; // rsi
  _QWORD *v36; // r14
  _QWORD *v37; // r13
  __int64 v38; // rdi
  __int64 v40; // rsi
  unsigned __int8 *v41; // rsi
  __int64 v42; // rax
  __int64 v43; // rdx
  __int64 v44; // [rsp+0h] [rbp-150h]
  __int64 v45; // [rsp+10h] [rbp-140h]
  __int64 v46; // [rsp+20h] [rbp-130h]
  __int64 *v47; // [rsp+28h] [rbp-128h]
  __int64 v48; // [rsp+30h] [rbp-120h]
  __int64 v49; // [rsp+38h] [rbp-118h]
  __int64 *v50; // [rsp+38h] [rbp-118h]
  unsigned int v51; // [rsp+44h] [rbp-10Ch] BYREF
  __int64 v52; // [rsp+48h] [rbp-108h] BYREF
  __int64 v53[4]; // [rsp+50h] [rbp-100h] BYREF
  __int16 v54; // [rsp+70h] [rbp-E0h]
  _BYTE *v55; // [rsp+80h] [rbp-D0h] BYREF
  __int64 v56; // [rsp+88h] [rbp-C8h]
  _BYTE v57[64]; // [rsp+90h] [rbp-C0h] BYREF
  __int64 *v58; // [rsp+D0h] [rbp-80h] BYREF
  __int64 v59; // [rsp+D8h] [rbp-78h]
  _BYTE v60[112]; // [rsp+E0h] [rbp-70h] BYREF

  v7 = *a1;
  if ( v7 == 40 )
  {
    v8 = -32 - 32LL * (unsigned int)sub_B491D0((__int64)a1);
  }
  else
  {
    v8 = -32;
    if ( v7 != 85 )
    {
      v8 = -96;
      if ( v7 != 34 )
        BUG();
    }
  }
  if ( (a1[7] & 0x80u) != 0 )
  {
    v9 = sub_BD2BC0((__int64)a1);
    v11 = v9 + v10;
    v12 = 0;
    if ( (a1[7] & 0x80u) != 0 )
      v12 = sub_BD2BC0((__int64)a1);
    if ( (unsigned int)((v11 - v12) >> 4) )
    {
      if ( (a1[7] & 0x80u) == 0 )
        BUG();
      v13 = *(_DWORD *)(sub_BD2BC0((__int64)a1) + 8);
      if ( (a1[7] & 0x80u) == 0 )
        BUG();
      v14 = sub_BD2BC0((__int64)a1);
      v8 -= 32LL * (unsigned int)(*(_DWORD *)(v14 + v15 - 4) - v13);
    }
  }
  v16 = *((_DWORD *)a1 + 1);
  v17 = &a1[v8];
  v58 = (__int64 *)v60;
  v18 = 0;
  v59 = 0x800000000LL;
  v19 = (__int64 *)v60;
  v20 = 32LL * (v16 & 0x7FFFFFF);
  v21 = &a1[-v20];
  v22 = v8 + v20;
  v23 = v22 >> 5;
  if ( (unsigned __int64)v22 > 0x100 )
  {
    sub_C8D5F0((__int64)&v58, v60, v22 >> 5, 8u, a5, a6);
    v18 = v59;
    v19 = &v58[(unsigned int)v59];
  }
  if ( v21 != v17 )
  {
    do
    {
      if ( v19 )
        *v19 = *(_QWORD *)v21;
      v21 += 32;
      ++v19;
    }
    while ( v17 != v21 );
    v18 = v59;
  }
  v55 = v57;
  LODWORD(v59) = v18 + v23;
  v56 = 0x100000000LL;
  sub_B56970((__int64)a1, (__int64)&v55);
  v24 = v55;
  v54 = 257;
  v47 = v58;
  v46 = (unsigned int)v59;
  v48 = *((_QWORD *)a1 - 4);
  v49 = *((_QWORD *)a1 + 10);
  v25 = &v55[56 * (unsigned int)v56];
  if ( v55 == v25 )
  {
    v27 = 0;
  }
  else
  {
    v26 = v55;
    v27 = 0;
    do
    {
      v28 = *((_QWORD *)v26 + 5) - *((_QWORD *)v26 + 4);
      v26 += 56;
      v27 += v28 >> 3;
    }
    while ( v25 != v26 );
  }
  v29 = (unsigned int)(v59 + v27 + 1);
  v45 = (unsigned int)v56;
  LOBYTE(v17) = 16 * (_DWORD)v56 != 0;
  v30 = sub_BD2CC0(88, ((unsigned __int64)(unsigned int)(16 * v56) << 32) | v29);
  v31 = (__int64)v30;
  if ( v30 )
  {
    v44 = (__int64)v24;
    v32 = (__int64)v30;
    sub_B44260((__int64)v30, **(_QWORD **)(v49 + 16), 56, ((_DWORD)v17 << 28) | v29 & 0x7FFFFFF, 0, 0);
    *(_QWORD *)(v31 + 72) = 0;
    sub_B4A290(v31, v49, v48, v47, v46, (__int64)v53, v44, v45);
  }
  else
  {
    v32 = 0;
  }
  v33 = (__int64 *)(v31 + 48);
  *(_WORD *)(v31 + 2) = *((_WORD *)a1 + 1) & 0xFFC | *(_WORD *)(v31 + 2) & 0xF003;
  *(_QWORD *)(v31 + 72) = *((_QWORD *)a1 + 9);
  v34 = *((_QWORD *)a1 + 6);
  v53[0] = v34;
  if ( !v34 )
  {
    if ( v33 == v53 )
      goto LABEL_26;
    v40 = *(_QWORD *)(v31 + 48);
    if ( !v40 )
      goto LABEL_26;
LABEL_42:
    v50 = v33;
    sub_B91220((__int64)v33, v40);
    v33 = v50;
    goto LABEL_43;
  }
  sub_B96E90((__int64)v53, v34, 1);
  v33 = (__int64 *)(v31 + 48);
  if ( (__int64 *)(v31 + 48) == v53 )
  {
    if ( v53[0] )
      sub_B91220((__int64)v53, v53[0]);
    goto LABEL_26;
  }
  v40 = *(_QWORD *)(v31 + 48);
  if ( v40 )
    goto LABEL_42;
LABEL_43:
  v41 = (unsigned __int8 *)v53[0];
  *(_QWORD *)(v31 + 48) = v53[0];
  if ( v41 )
    sub_B976B0((__int64)v53, v41, (__int64)v33);
LABEL_26:
  sub_B47C00(v32, (__int64)a1, 0, 0);
  v35 = (__int64)&v52;
  if ( (unsigned __int8)sub_B92100(v32, &v52) )
  {
    v42 = sub_BD5C60(v32);
    v43 = 0;
    v53[0] = v42;
    if ( v52 == (unsigned int)v52 )
    {
      v51 = v52;
      v43 = sub_B8C150(v53, &v51, 1, 0);
    }
    v35 = 2;
    sub_B99FD0(v32, 2u, v43);
  }
  v36 = v55;
  v37 = &v55[56 * (unsigned int)v56];
  if ( v55 != (_BYTE *)v37 )
  {
    do
    {
      v38 = *(v37 - 3);
      v37 -= 7;
      if ( v38 )
      {
        v35 = v37[6] - v38;
        j_j___libc_free_0(v38, v35);
      }
      if ( (_QWORD *)*v37 != v37 + 2 )
      {
        v35 = v37[2] + 1LL;
        j_j___libc_free_0(*v37, v35);
      }
    }
    while ( v36 != v37 );
    v37 = v55;
  }
  if ( v37 != (_QWORD *)v57 )
    _libc_free(v37, v35);
  if ( v58 != (__int64 *)v60 )
    _libc_free(v58, v35);
  return v31;
}
