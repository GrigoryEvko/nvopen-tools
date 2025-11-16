// Function: sub_1308BE0
// Address: 0x1308be0
//
unsigned __int64 __fastcall sub_1308BE0(__int64 a1, unsigned __int64 a2, unsigned __int64 a3, int a4, int a5, int a6)
{
  __int64 v6; // r12
  unsigned __int64 v7; // rax
  unsigned __int8 v8; // dl
  __int64 v9; // rdx
  unsigned __int64 v10; // rbx
  __int64 v11; // r14
  unsigned __int64 v12; // r10
  unsigned __int64 v13; // rax
  unsigned int v14; // ecx
  unsigned __int64 v15; // r15
  unsigned __int64 v16; // rsi
  void *v17; // rax
  unsigned __int64 v19; // rdx
  unsigned __int64 v20; // r13
  void **v21; // rax
  __int64 v22; // rdx
  void **v23; // rdi
  __int64 v24; // rax
  void **v25; // r15
  void **v26; // rbx
  void *v27; // rdi
  __int64 v28; // rax
  __int64 *v29; // r13
  __int64 v30; // rax
  __int64 v31; // rdx
  char v32; // cl
  __int64 v33; // rax
  char v34; // cl
  unsigned __int64 v35; // rcx
  __int64 v36; // rax
  char v37; // cl
  __int64 v38; // rdx
  char v39; // cl
  unsigned int v40; // [rsp+4h] [rbp-DCh]
  __int64 *v41; // [rsp+10h] [rbp-D0h]
  unsigned __int64 v42; // [rsp+20h] [rbp-C0h]
  unsigned __int64 v44; // [rsp+38h] [rbp-A8h]
  unsigned __int8 v45; // [rsp+47h] [rbp-99h]
  unsigned __int64 v46; // [rsp+48h] [rbp-98h]
  unsigned __int64 v47; // [rsp+48h] [rbp-98h]
  unsigned __int64 v48; // [rsp+48h] [rbp-98h]
  unsigned __int64 v49; // [rsp+48h] [rbp-98h]
  unsigned __int64 v50; // [rsp+48h] [rbp-98h]
  unsigned __int64 n; // [rsp+50h] [rbp-90h]
  unsigned int v53; // [rsp+5Ch] [rbp-84h]
  __int64 v54; // [rsp+60h] [rbp-80h]
  unsigned __int64 v56; // [rsp+70h] [rbp-70h]
  char v58[8]; // [rsp+80h] [rbp-60h] BYREF
  __int64 v59; // [rsp+88h] [rbp-58h]
  __int64 v60; // [rsp+90h] [rbp-50h]
  __int64 v61; // [rsp+98h] [rbp-48h]
  __int64 v62; // [rsp+A0h] [rbp-40h]

  v6 = __readfsqword(0) - 2664;
  if ( __readfsbyte(0xFFFFF8C8) )
  {
    v6 = sub_1313D30(v6, 0);
    if ( !v6 )
      return 0;
  }
  if ( *(char *)(v6 + 1) > 0 )
    return 0;
  v7 = (1LL << a4) & 0xFFFFFFFFFFFFFFFELL;
  if ( !v7 )
  {
    if ( a3 > 0x1000 )
    {
      if ( a3 > 0x7000000000000000LL )
        return 0;
      _BitScanReverse64((unsigned __int64 *)&v36, 2 * a3 - 1);
      if ( (unsigned __int64)(int)v36 < 7 )
        LOBYTE(v36) = 7;
      n = -(1LL << ((unsigned __int8)v36 - 3)) & ((1LL << ((unsigned __int8)v36 - 3)) + a3 - 1);
    }
    else
    {
      n = qword_505FA40[byte_5060800[(a3 + 7) >> 3]];
    }
    goto LABEL_6;
  }
  if ( a3 > 0x3800 || v7 > 0x1000 )
  {
    if ( v7 > 0x7000000000000000LL )
      return 0;
    if ( a3 > 0x4000 )
    {
      if ( a3 > 0x7000000000000000LL )
        return 0;
      _BitScanReverse64((unsigned __int64 *)&v31, 2 * a3 - 1);
      if ( (unsigned __int64)(int)v31 < 7 )
        LOBYTE(v31) = 7;
      n = -(1LL << ((unsigned __int8)v31 - 3)) & ((1LL << ((unsigned __int8)v31 - 3)) + a3 - 1);
      if ( a3 > n
        || __CFADD__(n, unk_50607C0 + ((v7 + 4095) & 0xFFFFFFFFFFFFF000LL) - 4096)
        || n - 1 > 0x6FFFFFFFFFFFFFFFLL )
      {
        return 0;
      }
      v32 = 7;
      _BitScanReverse64((unsigned __int64 *)&v33, 2 * n - 1);
      if ( (unsigned int)v33 >= 7 )
        v32 = v33;
      v34 = v32 - 3;
      if ( (unsigned int)v33 < 6 )
        LODWORD(v33) = 6;
      v53 = ((((-1LL << v34) & (n - 1)) >> v34) & 3) + 4 * v33 - 23;
      goto LABEL_9;
    }
LABEL_33:
    if ( ((v7 + 4095) & 0xFFFFFFFFFFFFF000LL) + unk_50607C0 + 12288 <= 0x3FFF )
      return 0;
    v53 = 36;
    n = 0x4000;
    goto LABEL_9;
  }
  v19 = -(__int64)v7 & (v7 + a3 - 1);
  if ( v19 > 0x1000 )
  {
    _BitScanReverse64(&v35, 2 * v19 - 1);
    n = -(1LL << ((unsigned __int8)v35 - 3)) & (v19 + (1LL << ((unsigned __int8)v35 - 3)) - 1);
  }
  else
  {
    n = qword_505FA40[byte_5060800[(v19 + 7) >> 3]];
  }
  if ( n > 0x3FFF )
    goto LABEL_33;
LABEL_6:
  if ( n - 1 > 0x6FFFFFFFFFFFFFFFLL )
    return 0;
  if ( n > 0x1000 )
  {
    v37 = 7;
    _BitScanReverse64((unsigned __int64 *)&v38, 2 * n - 1);
    if ( (unsigned int)v38 >= 7 )
      v37 = v38;
    v39 = v37 - 3;
    if ( (unsigned int)v38 < 6 )
      LODWORD(v38) = 6;
    v53 = ((((-1LL << v39) & (n - 1)) >> v39) & 3) + 4 * v38 - 23;
  }
  else
  {
    v53 = byte_5060800[(n + 7) >> 3];
  }
LABEL_9:
  v8 = unk_4F96994;
  if ( !unk_4F96994 )
    v8 = (a4 & 0x40) != 0;
  v45 = v8;
  if ( v53 > 0x23 )
    v56 = 0;
  else
    v56 = *((unsigned int *)&unk_5260DE0 + 10 * v53 + 4);
  if ( a2 )
  {
    v54 = 0;
    v9 = ((unsigned int)a4 >> 20) - 1;
    v41 = &qword_50579C0[v9];
    v10 = 0;
    v11 = 0;
    while ( 1 )
    {
      v12 = a2 - v10;
      if ( a2 - v10 < v56 || v53 > 0x23 )
      {
        v14 = v53;
        v15 = 0;
        v16 = 0;
        if ( v53 >= unk_5060A18 )
          goto LABEL_22;
        goto LABEL_36;
      }
      if ( !v54 )
      {
        if ( (a4 & 0xFFF00000) != 0 )
        {
          v54 = *v41;
          if ( *v41 )
            goto LABEL_19;
          v40 = ((unsigned int)a4 >> 20) - 1;
          v28 = sub_1300B80(v6, v40, (__int64)&off_49E8000);
          v12 = a2 - v10;
          v54 = v28;
          if ( v28 )
            goto LABEL_19;
          if ( unk_505F9B8 <= v40 )
            return v10;
        }
        v47 = v12;
        v24 = sub_1302E60(v6, 0);
        v12 = v47;
        v54 = v24;
        if ( !v24 )
          return v10;
      }
LABEL_19:
      v46 = v12;
      v13 = sub_1316B20(v6, v54, v53, a1 + 8 * v10, v12 - v12 % v56, v45);
      v12 = v46;
      v10 += v13;
      v15 = v13;
      if ( unk_5060A18 <= v53 || v46 <= v13 )
        goto LABEL_21;
LABEL_36:
      if ( v11 )
        goto LABEL_37;
      if ( (a4 & 0xFFF00) != 0 )
      {
        if ( (a4 & 0xFFF00) == 0x100 )
          goto LABEL_21;
        if ( ((a4 >> 8) & 0xFFF) != 0 )
        {
          if ( ((a4 >> 8) & 0xFFF) == 1 )
            goto LABEL_21;
          v29 = (__int64 *)(*(_QWORD *)&dword_5060A08 + 8LL * (((a4 >> 8) & 0xFFFu) - 2));
          v11 = *v29;
          if ( !*v29 )
          {
            sub_130ACF0((unsigned int)"<jemalloc>: invalid tcache id (%u).\n", ((a4 >> 8) & 0xFFF) - 2, v9, v14, a5, a6);
            abort();
          }
          if ( v11 == 1 )
          {
            v50 = v12;
            v30 = sub_1311F90(v6);
            v12 = v50;
            *v29 = v30;
            v11 = v30;
            if ( !v30 )
              goto LABEL_21;
          }
          goto LABEL_50;
        }
      }
      if ( *(_BYTE *)v6 )
      {
        v11 = v6 + 856;
LABEL_50:
        v11 += 24LL * v53 + 8;
LABEL_37:
        v44 = v12;
        v20 = (unsigned __int16)(*(_WORD *)(v11 + 20) - *(_QWORD *)v11) >> 3;
        if ( v20 > v12 - v15 )
          v20 = v12 - v15;
        v21 = (void **)memcpy((void *)(a1 + 8 * v10), *(const void **)v11, 8 * v20);
        v22 = *(_QWORD *)v11 + 8 * v20;
        v23 = v21;
        LOWORD(v21) = *(_WORD *)(v11 + 20);
        *(_QWORD *)v11 = v22;
        v12 = v44;
        if ( (unsigned __int16)((unsigned __int16)((_WORD)v21 - v22) >> 3) < (unsigned __int16)((unsigned __int16)((_WORD)v21 - *(_WORD *)(v11 + 16)) >> 3) )
          *(_WORD *)(v11 + 16) = v22;
        *(_QWORD *)(v11 + 8) += v20;
        if ( v45 )
        {
          if ( v20 )
          {
            v49 = v15;
            v42 = v20 + v10;
            v25 = (void **)(a1 + 8 * (v20 + v10));
            v26 = v23;
            do
            {
              v27 = *v26++;
              memset(v27, 0, n);
            }
            while ( v26 != v25 );
            v15 = v49;
            v12 = v44;
            v10 = v42;
          }
        }
        else
        {
          v10 += v20;
        }
        v15 += v20;
      }
LABEL_21:
      v16 = v15 * n;
LABEL_22:
      v9 = *(_QWORD *)(v6 + 824);
      v58[0] = 1;
      v59 = v6 + 824;
      v60 = v6 + 8;
      v61 = v6 + 16;
      v62 = v6 + 832;
      *(_QWORD *)(v6 + 824) = v16 + v9;
      if ( v16 >= *(_QWORD *)(v6 + 16) - v9 )
      {
        v48 = v12;
        sub_13133F0(v6, v58);
        v12 = v48;
      }
      if ( v15 >= v12 )
      {
        if ( a2 <= v10 )
          return v10;
      }
      else
      {
        v17 = sub_1307610(a3, a4);
        if ( !v17 )
          return v10;
        *(_QWORD *)(a1 + 8 * v10++) = v17;
        if ( a2 <= v10 )
          return v10;
      }
    }
  }
  return 0;
}
