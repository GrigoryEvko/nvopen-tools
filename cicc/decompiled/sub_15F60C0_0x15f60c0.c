// Function: sub_15F60C0
// Address: 0x15f60c0
//
__int64 __fastcall sub_15F60C0(__int64 a1, __int64 *a2, __int64 a3, __int64 a4)
{
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // r12
  int v10; // r12d
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rcx
  char *v14; // r12
  __int64 v15; // rax
  char *v16; // rsi
  __int64 v17; // rax
  __int64 *v18; // r15
  __int64 *v19; // rdx
  __int64 *v20; // rax
  __int64 v21; // r9
  int v22; // r8d
  __int64 v23; // rax
  __int64 v24; // r11
  __int64 v25; // rdx
  __int64 *v26; // rcx
  __int64 *v27; // rdx
  int v28; // esi
  __int64 v29; // rax
  __int64 v30; // r12
  __int64 v31; // r11
  __int64 v32; // r10
  __int64 v33; // rax
  __int64 *v34; // rsi
  __int64 v35; // rdi
  __int64 v36; // rdx
  int v37; // r8d
  __int64 v38; // r9
  _QWORD *v39; // r14
  __int16 v40; // dx
  __int64 v41; // rsi
  __int64 v43; // rsi
  __int64 v44; // rax
  __int64 v45; // [rsp+8h] [rbp-98h]
  __int64 v46; // [rsp+10h] [rbp-90h]
  __int64 v47; // [rsp+18h] [rbp-88h]
  __int64 v48; // [rsp+18h] [rbp-88h]
  __int64 v49; // [rsp+20h] [rbp-80h]
  __int64 v50; // [rsp+20h] [rbp-80h]
  __int64 v51; // [rsp+28h] [rbp-78h]
  __int64 v52; // [rsp+28h] [rbp-78h]
  __int64 v53; // [rsp+28h] [rbp-78h]
  unsigned int v54; // [rsp+30h] [rbp-70h]
  __int64 v55; // [rsp+30h] [rbp-70h]
  _QWORD v57[2]; // [rsp+40h] [rbp-60h] BYREF
  _QWORD v58[2]; // [rsp+50h] [rbp-50h] BYREF
  __int16 v59; // [rsp+60h] [rbp-40h]

  if ( *(char *)(a1 + 23) >= 0 )
    goto LABEL_8;
  v7 = sub_1648A40(a1);
  v9 = v7 + v8;
  if ( *(char *)(a1 + 23) >= 0 )
  {
    if ( (unsigned int)(v9 >> 4) )
LABEL_45:
      BUG();
LABEL_8:
    v13 = -24;
    goto LABEL_9;
  }
  if ( !(unsigned int)((v9 - sub_1648A40(a1)) >> 4) )
    goto LABEL_8;
  if ( *(char *)(a1 + 23) >= 0 )
    goto LABEL_45;
  v10 = *(_DWORD *)(sub_1648A40(a1) + 8);
  if ( *(char *)(a1 + 23) >= 0 )
    BUG();
  v11 = sub_1648A40(a1);
  v13 = -24 - 24LL * (unsigned int)(*(_DWORD *)(v11 + v12 - 4) - v10);
LABEL_9:
  v14 = (char *)(a1 + v13);
  v15 = 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF);
  v16 = (char *)(a1 - v15);
  v17 = v13 + v15;
  if ( v17 < 0 )
    sub_4262D8((__int64)"cannot create std::vector larger than max_size()");
  v18 = 0;
  v45 = 0x5555555555555558LL * (v17 >> 3);
  if ( 0xAAAAAAAAAAAAAAABLL * (v17 >> 3) )
    v18 = (__int64 *)sub_22077B0(0x5555555555555558LL * (v17 >> 3));
  if ( v16 == v14 )
  {
    v21 = 0;
    v22 = 1;
  }
  else
  {
    v19 = v18;
    v20 = (__int64 *)v16;
    do
    {
      if ( v19 )
        *v19 = *v20;
      v20 += 3;
      ++v19;
    }
    while ( v14 != (char *)v20 );
    v21 = (__int64)(0x5555555555555558LL * ((unsigned __int64)(v14 - v16 - 24) >> 3) + 8) >> 3;
    v22 = v21 + 1;
  }
  v51 = v21;
  v54 = v22;
  v23 = sub_1649960(a1);
  v24 = *(_QWORD *)(a1 - 24);
  v57[0] = v23;
  v59 = 261;
  v58[0] = v57;
  v57[1] = v25;
  v26 = &a2[7 * a3];
  if ( v26 == a2 )
  {
    v50 = v24;
    v48 = v51;
    v53 = *(_QWORD *)(*(_QWORD *)v24 + 24LL);
    v44 = sub_1648AB0(72, v54, (unsigned int)(16 * a3));
    v37 = v54;
    v32 = v53;
    v31 = v50;
    v38 = v48;
    v30 = v44;
    if ( v44 )
    {
      v33 = v48;
LABEL_25:
      v49 = v33;
      v52 = v31;
      v55 = v32;
      sub_15F1EA0(v30, **(_QWORD **)(v32 + 16), 54, v30 - 24 * v38 - 24, v37, a4);
      *(_QWORD *)(v30 + 56) = 0;
      sub_15F5B40(v30, v55, v52, v18, v49, (__int64)v58, a2, a3);
    }
  }
  else
  {
    v27 = a2;
    v28 = 0;
    do
    {
      v29 = v27[5] - v27[4];
      v27 += 7;
      v28 += v29 >> 3;
    }
    while ( v26 != v27 );
    v46 = *(_QWORD *)(*(_QWORD *)v24 + 24LL);
    v47 = v24;
    v30 = sub_1648AB0(72, v54 + v28, (unsigned int)(16 * a3));
    v31 = v47;
    v32 = v46;
    if ( v30 )
    {
      v33 = v51;
      v34 = a2;
      LODWORD(v35) = 0;
      do
      {
        v36 = v34[5] - v34[4];
        v34 += 7;
        v35 = (unsigned int)(v36 >> 3) + (unsigned int)v35;
      }
      while ( &a2[7 * a3] != v34 );
      v37 = v35 + v54;
      v38 = v35 + v51;
      goto LABEL_25;
    }
  }
  v39 = (_QWORD *)(v30 + 48);
  v40 = *(_WORD *)(a1 + 18) & 3 | *(_WORD *)(v30 + 18) & 0xFFFC;
  *(_WORD *)(v30 + 18) = v40;
  *(_WORD *)(v30 + 18) = v40 & 0x8000 | v40 & 3 | (4 * ((*(_WORD *)(a1 + 18) >> 2) & 0xDFFF));
  *(_BYTE *)(v30 + 17) = *(_BYTE *)(a1 + 17) & 0xFE | *(_BYTE *)(v30 + 17) & 1;
  *(_QWORD *)(v30 + 56) = *(_QWORD *)(a1 + 56);
  v41 = *(_QWORD *)(a1 + 48);
  v58[0] = v41;
  if ( !v41 )
  {
    if ( v39 == v58 || !*(_QWORD *)(v30 + 48) )
      goto LABEL_30;
LABEL_34:
    sub_161E7C0(v30 + 48);
    goto LABEL_35;
  }
  sub_1623A60(v58, v41, 2);
  if ( v39 == v58 )
  {
    if ( v58[0] )
      sub_161E7C0(v58);
    goto LABEL_30;
  }
  if ( *(_QWORD *)(v30 + 48) )
    goto LABEL_34;
LABEL_35:
  v43 = v58[0];
  *(_QWORD *)(v30 + 48) = v58[0];
  if ( v43 )
    sub_1623210(v58, v43, v30 + 48);
LABEL_30:
  if ( v18 )
    j_j___libc_free_0(v18, v45);
  return v30;
}
