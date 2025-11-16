// Function: sub_B4AD00
// Address: 0xb4ad00
//
__int64 __fastcall sub_B4AD00(unsigned __int8 *a1, __int64 a2, __int64 a3, __int64 a4, unsigned __int16 a5)
{
  int v6; // edx
  __int64 v7; // r14
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // r13
  __int64 v11; // rax
  int v12; // r13d
  __int64 v13; // rax
  __int64 v14; // rdx
  unsigned __int8 *v15; // r15
  __int64 v16; // rax
  unsigned __int8 *v17; // rsi
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 *v20; // r13
  __int64 *v21; // rdx
  __int64 *v22; // rax
  int v23; // r14d
  __int64 v24; // rax
  __int64 v25; // r15
  __int64 v26; // rax
  __int64 v27; // rdx
  __int64 v28; // rdx
  __int64 v29; // rax
  int v30; // ecx
  __int64 v31; // rsi
  __int64 v32; // rax
  __int64 v33; // rax
  __int64 v34; // r14
  _QWORD *v35; // r15
  __int64 v36; // rsi
  __int64 v37; // rcx
  __int64 v38; // r8
  __int64 v39; // r9
  __int64 v41; // rsi
  __int64 v42; // [rsp+8h] [rbp-B8h]
  __int64 v43; // [rsp+10h] [rbp-B0h]
  int v45; // [rsp+28h] [rbp-98h]
  unsigned __int16 v46; // [rsp+2Eh] [rbp-92h]
  __int64 v47; // [rsp+30h] [rbp-90h]
  __int64 v48; // [rsp+38h] [rbp-88h]
  __int64 v49; // [rsp+40h] [rbp-80h]
  int v52; // [rsp+50h] [rbp-70h]
  _QWORD v53[4]; // [rsp+60h] [rbp-60h] BYREF
  __int16 v54; // [rsp+80h] [rbp-40h]

  v6 = *a1;
  if ( v6 == 40 )
  {
    v7 = -32 - 32LL * (unsigned int)sub_B491D0((__int64)a1);
  }
  else
  {
    v7 = -32;
    if ( v6 != 85 )
    {
      v7 = -96;
      if ( v6 != 34 )
        BUG();
    }
  }
  if ( (a1[7] & 0x80u) != 0 )
  {
    v8 = sub_BD2BC0(a1);
    v10 = v8 + v9;
    v11 = 0;
    if ( (a1[7] & 0x80u) != 0 )
      v11 = sub_BD2BC0(a1);
    if ( (unsigned int)((v10 - v11) >> 4) )
    {
      if ( (a1[7] & 0x80u) == 0 )
        BUG();
      v12 = *(_DWORD *)(sub_BD2BC0(a1) + 8);
      if ( (a1[7] & 0x80u) == 0 )
        BUG();
      v13 = sub_BD2BC0(a1);
      v7 -= 32LL * (unsigned int)(*(_DWORD *)(v13 + v14 - 4) - v12);
    }
  }
  v15 = &a1[v7];
  v16 = 32LL * (*((_DWORD *)a1 + 1) & 0x7FFFFFF);
  v17 = &a1[-v16];
  v18 = v7 + v16;
  v19 = v18 >> 5;
  if ( v18 < 0 )
    sub_4262D8((__int64)"cannot create std::vector larger than max_size()");
  v20 = 0;
  v42 = 8 * v19;
  if ( v19 )
    v20 = (__int64 *)sub_22077B0(8 * v19);
  if ( v17 == v15 )
  {
    v43 = 0;
    v23 = 3;
  }
  else
  {
    v21 = v20;
    v22 = (__int64 *)v17;
    do
    {
      if ( v21 )
        *v21 = *v22;
      v22 += 4;
      ++v21;
    }
    while ( v22 != (__int64 *)v15 );
    v43 = (__int64)(8 * ((unsigned __int64)((char *)v22 - (char *)v17 - 32) >> 5) + 8) >> 3;
    v23 = v43 + 3;
  }
  v24 = sub_BD5D20(a1);
  v25 = *((_QWORD *)a1 + 10);
  v53[0] = v24;
  v26 = *((_QWORD *)a1 - 8);
  v53[1] = v27;
  v28 = a2;
  v47 = v26;
  v29 = *((_QWORD *)a1 - 12);
  v54 = 261;
  v30 = 0;
  v48 = v29;
  v49 = *((_QWORD *)a1 - 4);
  v46 = a5;
  v31 = a2 + 56 * a3;
  if ( a2 != v31 )
  {
    do
    {
      v32 = *(_QWORD *)(v28 + 40) - *(_QWORD *)(v28 + 32);
      v28 += 56;
      v30 += v32 >> 3;
    }
    while ( v31 != v28 );
  }
  v52 = v23 + v30;
  v33 = sub_BD2CC0(88, ((unsigned __int64)(unsigned int)(16 * a3) << 32) | (unsigned int)(v23 + v30));
  v34 = v33;
  if ( v33 )
  {
    LOBYTE(v45) = 16 * (_DWORD)a3 != 0;
    sub_B44260(v33, **(_QWORD **)(v25 + 16), 5, (v45 << 28) | v52 & 0x7FFFFFF, a4, v46);
    *(_QWORD *)(v34 + 72) = 0;
    sub_B4A9C0(v34, v25, v49, v48, v47, (__int64)v53, v20, v43, a2, a3);
  }
  v35 = (_QWORD *)(v34 + 48);
  *(_WORD *)(v34 + 2) = *((_WORD *)a1 + 1) & 0xFFC | *(_WORD *)(v34 + 2) & 0xF003;
  *(_BYTE *)(v34 + 1) = a1[1] & 0xFE | *(_BYTE *)(v34 + 1) & 1;
  *(_QWORD *)(v34 + 72) = *((_QWORD *)a1 + 9);
  v36 = *((_QWORD *)a1 + 6);
  v53[0] = v36;
  if ( !v36 )
  {
    if ( v35 == v53 || !*(_QWORD *)(v34 + 48) )
      goto LABEL_27;
LABEL_33:
    sub_B91220(v34 + 48);
    goto LABEL_34;
  }
  sub_B96E90(v53, v36, 1);
  if ( v35 == v53 )
  {
    if ( v53[0] )
      sub_B91220(v53);
    goto LABEL_27;
  }
  if ( *(_QWORD *)(v34 + 48) )
    goto LABEL_33;
LABEL_34:
  v41 = v53[0];
  *(_QWORD *)(v34 + 48) = v53[0];
  if ( v41 )
    sub_B976B0(v53, v41, v34 + 48, v37, v38, v39);
LABEL_27:
  if ( v20 )
    j_j___libc_free_0(v20, v42);
  return v34;
}
