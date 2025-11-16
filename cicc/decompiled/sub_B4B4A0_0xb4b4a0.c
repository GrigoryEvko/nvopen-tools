// Function: sub_B4B4A0
// Address: 0xb4b4a0
//
__int64 __fastcall sub_B4B4A0(unsigned __int8 *a1, __int64 a2, __int64 a3, __int64 a4, unsigned __int16 a5)
{
  unsigned __int8 *v6; // rbx
  int v7; // edx
  __int64 v8; // r13
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // r15
  __int64 v12; // rax
  int v13; // r15d
  __int64 v14; // rax
  __int64 v15; // rdx
  unsigned __int8 *v16; // r15
  __int64 v17; // rax
  unsigned __int8 *v18; // r12
  __int64 v19; // rax
  __int64 v20; // rdx
  __int64 *v21; // r13
  __int64 *v22; // rdx
  __int64 *v23; // rax
  __int64 v24; // rax
  __int64 v25; // r8
  __int64 v26; // rdx
  _BYTE *v27; // rcx
  __int64 v28; // rdx
  __int64 v29; // r13
  __int64 v30; // rbx
  _QWORD *v31; // rax
  __int64 v32; // r12
  int v33; // ecx
  _QWORD *v34; // r15
  int v35; // esi
  int v36; // ecx
  __int64 v37; // rdx
  __int64 v38; // rdi
  __int64 v39; // rax
  unsigned __int64 v40; // rsi
  __int64 v41; // rax
  __int64 v42; // r14
  _QWORD *v43; // r12
  _BYTE *v44; // rsi
  __int64 v45; // rcx
  __int64 v46; // r8
  __int64 v47; // r9
  _BYTE *v49; // rsi
  __int64 v50; // [rsp+8h] [rbp-168h]
  unsigned int v51; // [rsp+10h] [rbp-160h]
  _BYTE *v52; // [rsp+18h] [rbp-158h]
  unsigned int v53; // [rsp+30h] [rbp-140h]
  int v54; // [rsp+34h] [rbp-13Ch]
  unsigned __int16 v56; // [rsp+40h] [rbp-130h]
  __int64 v57; // [rsp+48h] [rbp-128h]
  _QWORD *v58; // [rsp+48h] [rbp-128h]
  __int64 v59; // [rsp+50h] [rbp-120h]
  __int64 *v61; // [rsp+60h] [rbp-110h]
  __int64 v62; // [rsp+60h] [rbp-110h]
  __int64 v64; // [rsp+70h] [rbp-100h]
  _QWORD v65[4]; // [rsp+80h] [rbp-F0h] BYREF
  __int16 v66; // [rsp+A0h] [rbp-D0h]
  _BYTE *v67; // [rsp+B0h] [rbp-C0h] BYREF
  __int64 v68; // [rsp+B8h] [rbp-B8h]
  _BYTE v69[176]; // [rsp+C0h] [rbp-B0h] BYREF

  v6 = a1;
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
  v50 = 8 * v20;
  if ( v20 )
    v21 = (__int64 *)sub_22077B0(8 * v20);
  if ( v18 == v16 )
  {
    v54 = 0;
    v64 = 0;
  }
  else
  {
    v22 = v21;
    v23 = (__int64 *)v18;
    do
    {
      if ( v22 )
        *v22 = *v23;
      v23 += 4;
      ++v22;
    }
    while ( v23 != (__int64 *)v16 );
    v64 = (__int64)(8 * ((unsigned __int64)((char *)v23 - (char *)v18 - 32) >> 5) + 8) >> 3;
    v54 = v64;
  }
  v24 = sub_BD5D20(a1);
  v25 = *((unsigned int *)a1 + 22);
  v67 = v69;
  v65[0] = v24;
  v66 = 261;
  v65[1] = v26;
  v68 = 0x1000000000LL;
  if ( (_DWORD)v25 )
  {
    v27 = v69;
    v61 = v21;
    v28 = 0;
    v29 = v25;
    v30 = 1;
    v31 = &v67;
    v32 = *(_QWORD *)&a1[-32 * v25 - 32];
    while ( 1 )
    {
      *(_QWORD *)&v27[8 * v28] = v32;
      v33 = v68;
      v28 = (unsigned int)(v68 + 1);
      LODWORD(v68) = v68 + 1;
      if ( v29 == v30 )
        break;
      v32 = *(_QWORD *)&a1[32 * (v30 - *((unsigned int *)a1 + 22)) - 32];
      if ( v28 + 1 > (unsigned __int64)HIDWORD(v68) )
      {
        v58 = v31;
        sub_C8D5F0(v31, v69, v28 + 1, 8);
        v28 = (unsigned int)v68;
        v31 = v58;
      }
      v27 = v67;
      ++v30;
    }
    v6 = a1;
    v34 = v31;
    v21 = v61;
    v35 = v33 + 3;
    v52 = v67;
    v25 = *((unsigned int *)a1 + 22);
    v51 = v28;
  }
  else
  {
    v51 = 0;
    v35 = 2;
    v34 = &v67;
    v52 = v69;
  }
  v36 = 0;
  v37 = a2;
  v56 = a5;
  v57 = *(_QWORD *)&v6[-32 * v25 - 64];
  v59 = *((_QWORD *)v6 - 4);
  v62 = *((_QWORD *)v6 + 10);
  v38 = a2 + 56 * a3;
  if ( v38 != a2 )
  {
    do
    {
      v39 = *(_QWORD *)(v37 + 40) - *(_QWORD *)(v37 + 32);
      v37 += 56;
      v36 += v39 >> 3;
    }
    while ( v38 != v37 );
  }
  v53 = v54 + v35 + v36;
  v40 = ((unsigned __int64)(unsigned int)(16 * a3) << 32) | v53;
  v41 = sub_BD2CC0(96, v40);
  v42 = v41;
  if ( v41 )
  {
    LOBYTE(v54) = 16 * (_DWORD)a3 != 0;
    sub_B44260(v41, **(_QWORD **)(v62 + 16), 11, (v54 << 28) | v53 & 0x7FFFFFF, a4, v56);
    *(_QWORD *)(v42 + 72) = 0;
    v40 = v62;
    sub_B4B130(v42, v62, v59, v57, (__int64)v52, v51, v21, v64, a2, a3, (__int64)v65);
  }
  if ( v67 != v69 )
    _libc_free(v67, v40);
  v43 = (_QWORD *)(v42 + 48);
  *(_WORD *)(v42 + 2) = *((_WORD *)v6 + 1) & 0xFFC | *(_WORD *)(v42 + 2) & 0xF003;
  *(_BYTE *)(v42 + 1) = v6[1] & 0xFE | *(_BYTE *)(v42 + 1) & 1;
  *(_QWORD *)(v42 + 72) = *((_QWORD *)v6 + 9);
  v44 = (_BYTE *)*((_QWORD *)v6 + 6);
  v67 = v44;
  if ( !v44 )
  {
    if ( v43 == v34 || !*(_QWORD *)(v42 + 48) )
      goto LABEL_36;
LABEL_42:
    sub_B91220(v42 + 48);
    goto LABEL_43;
  }
  sub_B96E90(v34, v44, 1);
  if ( v43 == v34 )
  {
    if ( v67 )
      sub_B91220(v34);
    goto LABEL_36;
  }
  if ( *(_QWORD *)(v42 + 48) )
    goto LABEL_42;
LABEL_43:
  v49 = v67;
  *(_QWORD *)(v42 + 48) = v67;
  if ( v49 )
    sub_B976B0(v34, v49, v42 + 48, v45, v46, v47);
LABEL_36:
  *(_DWORD *)(v42 + 88) = *((_DWORD *)v6 + 22);
  if ( v21 )
    j_j___libc_free_0(v21, v50);
  return v42;
}
