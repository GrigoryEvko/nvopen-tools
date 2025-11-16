// Function: sub_B4A5D0
// Address: 0xb4a5d0
//
__int64 __fastcall sub_B4A5D0(unsigned __int8 *a1, __int64 a2, __int64 a3, __int64 a4, unsigned __int16 a5)
{
  int v7; // edx
  __int64 v8; // r15
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // r14
  __int64 v12; // rax
  int v13; // r14d
  __int64 v14; // rax
  __int64 v15; // rdx
  unsigned __int8 *v16; // rsi
  __int64 v17; // rax
  unsigned __int8 *v18; // r8
  __int64 v19; // rax
  __int64 v20; // rdx
  __int64 *v21; // r14
  __int64 v22; // rax
  __int64 *v23; // rdx
  __int64 *v24; // rax
  int v25; // r15d
  __int64 v26; // rax
  __int64 v27; // rax
  __int64 v28; // rdx
  __int64 v29; // rsi
  __int64 v30; // rdx
  int v31; // ecx
  __int64 v32; // rax
  __int64 v33; // rax
  __int64 v34; // r15
  _QWORD *v35; // r13
  __int16 v36; // ax
  __int64 v37; // rsi
  __int64 v38; // rcx
  __int64 v39; // r8
  __int64 v40; // r9
  __int64 v42; // rsi
  __int64 v43; // [rsp+8h] [rbp-A8h]
  int v44; // [rsp+10h] [rbp-A0h]
  __int64 v45; // [rsp+20h] [rbp-90h]
  int v46; // [rsp+2Ch] [rbp-84h]
  __int64 v49; // [rsp+40h] [rbp-70h]
  unsigned __int8 *v50; // [rsp+48h] [rbp-68h]
  __int64 v51; // [rsp+48h] [rbp-68h]
  _QWORD v52[4]; // [rsp+50h] [rbp-60h] BYREF
  __int16 v53; // [rsp+70h] [rbp-40h]

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
    v9 = sub_BD2BC0(a1);
    v11 = v9 + v10;
    v12 = 0;
    if ( (a1[7] & 0x80u) != 0 )
      v12 = sub_BD2BC0(a1);
    if ( (unsigned int)((v11 - v12) >> 4) )
    {
      if ( (a1[7] & 0x80u) == 0 )
        BUG();
      v13 = *(_DWORD *)(sub_BD2BC0(a1) + 8);
      if ( (a1[7] & 0x80u) == 0 )
        BUG();
      v14 = sub_BD2BC0(a1);
      v8 -= 32LL * (unsigned int)(*(_DWORD *)(v14 + v15 - 4) - v13);
    }
  }
  v16 = &a1[v8];
  v17 = 32LL * (*((_DWORD *)a1 + 1) & 0x7FFFFFF);
  v18 = &a1[-v17];
  v19 = v8 + v17;
  v20 = v19 >> 5;
  if ( v19 < 0 )
    sub_4262D8((__int64)"cannot create std::vector larger than max_size()");
  v21 = 0;
  v43 = 8 * v20;
  if ( v20 )
  {
    v50 = v18;
    v22 = sub_22077B0(8 * v20);
    v16 = &a1[v8];
    v18 = v50;
    v21 = (__int64 *)v22;
  }
  if ( v18 == v16 )
  {
    v45 = 0;
    v25 = 1;
  }
  else
  {
    v23 = v21;
    v24 = (__int64 *)v18;
    do
    {
      if ( v23 )
        *v23 = *v24;
      v24 += 4;
      ++v23;
    }
    while ( v24 != (__int64 *)v16 );
    v45 = (__int64)(8 * ((unsigned __int64)((char *)v24 - (char *)v18 - 32) >> 5) + 8) >> 3;
    v25 = v45 + 1;
  }
  v52[0] = sub_BD5D20(a1);
  v26 = *((_QWORD *)a1 - 4);
  v53 = 261;
  v49 = v26;
  v27 = *((_QWORD *)a1 + 10);
  v52[1] = v28;
  v51 = v27;
  v29 = a2 + 56 * a3;
  if ( a2 == v29 )
  {
    v31 = 0;
  }
  else
  {
    v30 = a2;
    v31 = 0;
    do
    {
      v32 = *(_QWORD *)(v30 + 40) - *(_QWORD *)(v30 + 32);
      v30 += 56;
      v31 += v32 >> 3;
    }
    while ( v29 != v30 );
  }
  v44 = v25 + v31;
  v33 = sub_BD2CC0(88, ((unsigned __int64)(unsigned int)(16 * a3) << 32) | (unsigned int)(v25 + v31));
  v34 = v33;
  if ( v33 )
  {
    LOBYTE(v46) = 16 * (_DWORD)a3 != 0;
    sub_B44260(v33, **(_QWORD **)(v51 + 16), 56, (v46 << 28) | v44 & 0x7FFFFFF, a4, a5);
    *(_QWORD *)(v34 + 72) = 0;
    sub_B4A290(v34, v51, v49, v21, v45, (__int64)v52, a2, a3);
  }
  v35 = (_QWORD *)(v34 + 48);
  v36 = *((_WORD *)a1 + 1) & 3 | *(_WORD *)(v34 + 2) & 0xFFFC;
  *(_WORD *)(v34 + 2) = v36;
  *(_WORD *)(v34 + 2) = *((_WORD *)a1 + 1) & 0xFFC | v36 & 0xF003;
  *(_BYTE *)(v34 + 1) = a1[1] & 0xFE | *(_BYTE *)(v34 + 1) & 1;
  *(_QWORD *)(v34 + 72) = *((_QWORD *)a1 + 9);
  v37 = *((_QWORD *)a1 + 6);
  v52[0] = v37;
  if ( !v37 )
  {
    if ( v35 == v52 || !*(_QWORD *)(v34 + 48) )
      goto LABEL_28;
LABEL_34:
    sub_B91220(v34 + 48);
    goto LABEL_35;
  }
  sub_B96E90(v52, v37, 1);
  if ( v35 == v52 )
  {
    if ( v52[0] )
      sub_B91220(v52);
    goto LABEL_28;
  }
  if ( *(_QWORD *)(v34 + 48) )
    goto LABEL_34;
LABEL_35:
  v42 = v52[0];
  *(_QWORD *)(v34 + 48) = v52[0];
  if ( v42 )
    sub_B976B0(v52, v42, v34 + 48, v38, v39, v40);
LABEL_28:
  if ( v21 )
    j_j___libc_free_0(v21, v43);
  return v34;
}
