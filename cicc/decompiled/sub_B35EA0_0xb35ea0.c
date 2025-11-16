// Function: sub_B35EA0
// Address: 0xb35ea0
//
__int64 __fastcall sub_B35EA0(
        unsigned int **a1,
        __int64 a2,
        char *a3,
        __int64 a4,
        __int64 a5,
        unsigned int a6,
        unsigned __int16 a7)
{
  __int64 v10; // rdx
  __int64 v11; // rsi
  __int64 v12; // rax
  __int64 v13; // r13
  __int64 v14; // rax
  unsigned __int64 v15; // rdx
  int v16; // r8d
  __int64 v17; // r14
  __int64 v18; // rcx
  unsigned __int64 v19; // rax
  __int64 v20; // rdi
  __int64 v21; // rdx
  int v22; // esi
  __int64 v23; // rax
  unsigned __int64 v24; // rsi
  __int64 v25; // rax
  __int64 v26; // r15
  __int64 v27; // r11
  __int64 v28; // r14
  __int64 v29; // rsi
  unsigned int *v30; // r13
  __int64 v31; // rbx
  __int64 v32; // rdx
  __int64 *v33; // rsi
  unsigned __int64 v34; // rax
  char *v35; // rdi
  __int64 v37; // rsi
  __int64 v38; // rax
  __int64 v39; // r13
  __int64 v40; // rax
  unsigned __int64 v41; // rdx
  unsigned int *v42; // rdx
  unsigned int v43; // r13d
  __int64 *v44; // rax
  __int64 v45; // [rsp+8h] [rbp-E8h]
  unsigned int v46; // [rsp+10h] [rbp-E0h]
  __int64 v47; // [rsp+18h] [rbp-D8h]
  int v49; // [rsp+28h] [rbp-C8h]
  int v50; // [rsp+30h] [rbp-C0h]
  unsigned __int64 v51; // [rsp+40h] [rbp-B0h]
  __int64 v52; // [rsp+50h] [rbp-A0h] BYREF
  __int64 v53; // [rsp+58h] [rbp-98h]
  __int16 v54; // [rsp+70h] [rbp-80h]
  char *v55; // [rsp+80h] [rbp-70h] BYREF
  __int64 v56; // [rsp+88h] [rbp-68h]
  char v57[96]; // [rsp+90h] [rbp-60h] BYREF

  v55 = v57;
  v56 = 0x600000000LL;
  sub_B32F20((__int64)&v55, v57, a3, &a3[8 * a4]);
  if ( (unsigned __int8)sub_B6B000(*(unsigned int *)(a2 + 36)) )
  {
    v37 = *((unsigned __int8 *)a1 + 110);
    if ( BYTE1(a6) )
      v37 = a6;
    sub_E3F6F0(&v52, v37, v10, &v52);
    v38 = sub_B9B140(a1[9], v52, v53);
    v39 = sub_B9F6F0(a1[9], v38);
    v40 = (unsigned int)v56;
    v41 = (unsigned int)v56 + 1LL;
    if ( v41 > HIDWORD(v56) )
    {
      sub_C8D5F0(&v55, v57, v41, 8);
      v40 = (unsigned int)v56;
    }
    *(_QWORD *)&v55[8 * v40] = v39;
    LODWORD(v56) = v56 + 1;
  }
  v11 = (unsigned __int8)a7;
  if ( !HIBYTE(a7) )
    v11 = *((unsigned __int8 *)a1 + 109);
  sub_E3F8A0(&v52, v11);
  v12 = sub_B9B140(a1[9], v52, v53);
  v13 = sub_B9F6F0(a1[9], v12);
  v14 = (unsigned int)v56;
  v15 = (unsigned int)v56 + 1LL;
  if ( v15 > HIDWORD(v56) )
  {
    sub_C8D5F0(&v55, v57, v15, 8);
    v14 = (unsigned int)v56;
  }
  *(_QWORD *)&v55[8 * v14] = v13;
  v16 = v56;
  v17 = (__int64)a1[15];
  v54 = 257;
  v18 = (__int64)a1[14];
  v50 = (int)v55;
  v49 = v56 + 1;
  v19 = *(_QWORD *)(a2 + 24);
  LODWORD(v56) = v56 + 1;
  v51 = v19;
  v20 = v18 + 56 * v17;
  if ( v18 == v20 )
  {
    v22 = 0;
  }
  else
  {
    v21 = v18;
    v22 = 0;
    do
    {
      v23 = *(_QWORD *)(v21 + 40) - *(_QWORD *)(v21 + 32);
      v21 += 56;
      v22 += v23 >> 3;
    }
    while ( v20 != v21 );
  }
  v45 = v18;
  v46 = v16 + v22 + 2;
  LOBYTE(v13) = 16 * (_DWORD)v17 != 0;
  v24 = ((unsigned __int64)(unsigned int)(16 * v17) << 32) | v46;
  v25 = sub_BD2CC0(88, v24);
  v26 = v25;
  if ( v25 )
  {
    v27 = v17;
    v28 = v25;
    v47 = v27;
    sub_B44260(v25, **(_QWORD **)(v51 + 16), 56, ((_DWORD)v13 << 28) | v46 & 0x7FFFFFF, 0, 0);
    *(_QWORD *)(v26 + 72) = 0;
    v24 = v51;
    sub_B4A290(v26, v51, a2, v50, v49, (unsigned int)&v52, v45, v47);
  }
  else
  {
    v28 = 0;
  }
  if ( *((_BYTE *)a1 + 108) )
  {
    v44 = (__int64 *)sub_BD5C60(v28, v24);
    *(_QWORD *)(v26 + 72) = sub_A7A090((__int64 *)(v26 + 72), v44, -1, 72);
  }
  if ( (unsigned __int8)sub_920620(v28) )
  {
    v42 = a1[12];
    v43 = *((_DWORD *)a1 + 26);
    if ( v42 )
      sub_B99FD0(v26, 3, v42);
    sub_B45150(v26, v43);
  }
  v29 = v26;
  (*(void (__fastcall **)(unsigned int *, __int64, __int64, unsigned int *, unsigned int *))(*(_QWORD *)a1[11] + 16LL))(
    a1[11],
    v26,
    a5,
    a1[7],
    a1[8]);
  v30 = *a1;
  v31 = (__int64)&(*a1)[4 * *((unsigned int *)a1 + 2)];
  while ( (unsigned int *)v31 != v30 )
  {
    v32 = *((_QWORD *)v30 + 1);
    v29 = *v30;
    v30 += 4;
    sub_B99FD0(v26, v29, v32);
  }
  v33 = (__int64 *)sub_BD5C60(v28, v29);
  v34 = sub_A7A090((__int64 *)(v26 + 72), v33, -1, 72);
  v35 = v55;
  *(_QWORD *)(v26 + 72) = v34;
  if ( v35 != v57 )
    _libc_free(v35, v33);
  return v26;
}
