// Function: sub_2574FF0
// Address: 0x2574ff0
//
__int64 __fastcall sub_2574FF0(__int64 ***a1, unsigned __int8 *a2)
{
  __int64 v3; // rdx
  __int64 v5; // r13
  _QWORD *v6; // rax
  _QWORD *v7; // r14
  _QWORD *v8; // rax
  __int64 **v9; // r12
  __int64 v10; // rax
  _QWORD *v11; // rax
  __int64 **v12; // rax
  __int64 v13; // rax
  __int64 v14; // r9
  __int64 v15; // r14
  __int64 **v16; // r15
  unsigned int v17; // esi
  __int64 *v18; // rcx
  int v19; // r11d
  unsigned int v20; // edx
  __int64 v21; // rax
  unsigned __int8 *v22; // r10
  __int64 v23; // rax
  __int64 *v24; // r12
  __int64 v25; // rsi
  __int64 v26; // r8
  unsigned int v27; // esi
  int v28; // eax
  __int64 v29; // r9
  int v30; // eax
  __int64 v31; // rax
  unsigned __int8 **v32; // rax
  __int64 v33; // r8
  __int64 v34; // rcx
  int v35; // edx
  __int64 v36; // rax
  unsigned __int8 **v37; // rax
  int v38; // eax
  __int64 v39; // [rsp-60h] [rbp-60h]
  __int64 v40; // [rsp-58h] [rbp-58h] BYREF
  __int64 v41; // [rsp-50h] [rbp-50h] BYREF
  unsigned __int8 *v42; // [rsp-48h] [rbp-48h] BYREF
  int v43; // [rsp-40h] [rbp-40h]

  if ( (unsigned __int8)(*a2 - 34) > 0x33u )
    return 1;
  v3 = 0x8000000000041LL;
  if ( !_bittest64(&v3, (unsigned int)*a2 - 34) )
    return 1;
  v5 = sub_D5D560((__int64)a2, **a1);
  if ( !v5 )
  {
    if ( !(unsigned __int8)sub_D5CD30(a2, **a1) )
      return 1;
    v11 = (_QWORD *)sub_AA48A0(*((_QWORD *)a2 + 5));
    v12 = (__int64 **)sub_BCB2B0(v11);
    if ( !sub_D5D1D0(a2, **a1, v12) )
      return 1;
    v13 = sub_A777F0(0x50u, a1[2][16]);
    v15 = v13;
    if ( v13 )
    {
      *(_QWORD *)v13 = a2;
      *(_QWORD *)(v13 + 8) = 524;
      *(_WORD *)(v13 + 16) = 256;
      *(_QWORD *)(v13 + 56) = v13 + 72;
      *(_QWORD *)(v13 + 72) = 0;
      *(_QWORD *)(v13 + 64) = 0x100000000LL;
      *(_OWORD *)(v13 + 24) = 0;
      *(_OWORD *)(v13 + 40) = 0;
    }
    v16 = a1[1];
    v42 = a2;
    v43 = 0;
    v17 = *((_DWORD *)v16 + 32);
    if ( v17 )
    {
      v18 = v16[14];
      v19 = 1;
      v20 = (v17 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v21 = (__int64)&v18[2 * v20];
      v22 = *(unsigned __int8 **)v21;
      if ( a2 == *(unsigned __int8 **)v21 )
      {
LABEL_17:
        v23 = *(unsigned int *)(v21 + 8);
LABEL_18:
        v16[17][2 * v23 + 1] = v15;
        v24 = **a1;
        if ( v24
          && (!(unsigned __int8)sub_A73ED0((_QWORD *)a2 + 9, 23) && !(unsigned __int8)sub_B49560((__int64)a2, 23)
           || (unsigned __int8)sub_A73ED0((_QWORD *)a2 + 9, 4)
           || (unsigned __int8)sub_B49560((__int64)a2, 4)) )
        {
          v25 = *((_QWORD *)a2 - 4);
          if ( v25 )
          {
            if ( !*(_BYTE *)v25 && *(_QWORD *)(v25 + 24) == *((_QWORD *)a2 + 10) )
              sub_981210(*v24, v25, (unsigned int *)(v15 + 8));
          }
        }
        return 1;
      }
      while ( v22 != (unsigned __int8 *)-4096LL )
      {
        if ( v22 == (unsigned __int8 *)-8192LL && !v5 )
          v5 = v21;
        v14 = (unsigned int)(v19 + 1);
        v20 = (v17 - 1) & (v19 + v20);
        v21 = (__int64)&v18[2 * v20];
        v22 = *(unsigned __int8 **)v21;
        if ( a2 == *(unsigned __int8 **)v21 )
          goto LABEL_17;
        ++v19;
      }
      if ( !v5 )
        v5 = v21;
      v41 = v5;
      v38 = *((_DWORD *)v16 + 30);
      v16[13] = (__int64 *)((char *)v16[13] + 1);
      v35 = v38 + 1;
      if ( 4 * (v38 + 1) < 3 * v17 )
      {
        v34 = (__int64)a2;
        v33 = v17 >> 3;
        if ( v17 - *((_DWORD *)v16 + 31) - v35 > (unsigned int)v33 )
          goto LABEL_35;
        goto LABEL_34;
      }
    }
    else
    {
      v41 = 0;
      v16[13] = (__int64 *)((char *)v16[13] + 1);
    }
    v17 *= 2;
LABEL_34:
    sub_2574E40((__int64)(v16 + 13), v17);
    sub_2567D60((__int64)(v16 + 13), (__int64 *)&v42, &v41);
    v34 = (__int64)v42;
    v5 = v41;
    v35 = *((_DWORD *)v16 + 30) + 1;
LABEL_35:
    *((_DWORD *)v16 + 30) = v35;
    if ( *(_QWORD *)v5 != -4096 )
      --*((_DWORD *)v16 + 31);
    *(_QWORD *)v5 = v34;
    *(_DWORD *)(v5 + 8) = v43;
    v36 = *((unsigned int *)v16 + 36);
    if ( v36 + 1 > (unsigned __int64)*((unsigned int *)v16 + 37) )
    {
      sub_C8D5F0((__int64)(v16 + 17), v16 + 19, v36 + 1, 0x10u, v33, v14);
      v36 = *((unsigned int *)v16 + 36);
    }
    v37 = (unsigned __int8 **)&v16[17][2 * v36];
    *v37 = a2;
    v37[1] = 0;
    v23 = *((unsigned int *)v16 + 36);
    *((_DWORD *)v16 + 36) = v23 + 1;
    *(_DWORD *)(v5 + 8) = v23;
    goto LABEL_18;
  }
  v6 = (_QWORD *)sub_A777F0(0x50u, a1[2][16]);
  v7 = v6;
  if ( v6 )
  {
    *v6 = a2;
    v8 = v6 + 9;
    *(v8 - 8) = v5;
    *((_BYTE *)v8 - 56) = 0;
    *((_OWORD *)v8 - 3) = 0;
    *((_OWORD *)v8 - 2) = 0;
    *v8 = 0;
    v7[7] = v8;
    v7[8] = 0x100000000LL;
  }
  v9 = a1[1];
  v42 = a2;
  v43 = 0;
  if ( !(unsigned __int8)sub_2567D60((__int64)(v9 + 19), (__int64 *)&v42, &v40) )
  {
    v26 = v40;
    v41 = v40;
    v27 = *((_DWORD *)v9 + 44);
    v28 = *((_DWORD *)v9 + 42);
    v9[19] = (__int64 *)((char *)v9[19] + 1);
    v29 = 2 * v27;
    v30 = v28 + 1;
    if ( 4 * v30 >= 3 * v27 )
    {
      v27 *= 2;
    }
    else if ( v27 - *((_DWORD *)v9 + 43) - v30 > v27 >> 3 )
    {
LABEL_27:
      *((_DWORD *)v9 + 42) = v30;
      if ( *(_QWORD *)v26 != -4096 )
        --*((_DWORD *)v9 + 43);
      *(_QWORD *)v26 = v42;
      *(_DWORD *)(v26 + 8) = v43;
      v31 = *((unsigned int *)v9 + 48);
      if ( v31 + 1 > (unsigned __int64)*((unsigned int *)v9 + 49) )
      {
        v39 = v26;
        sub_C8D5F0((__int64)(v9 + 23), v9 + 25, v31 + 1, 0x10u, v26, v29);
        v31 = *((unsigned int *)v9 + 48);
        v26 = v39;
      }
      v32 = (unsigned __int8 **)&v9[23][2 * v31];
      *v32 = a2;
      v32[1] = 0;
      v10 = *((unsigned int *)v9 + 48);
      *((_DWORD *)v9 + 48) = v10 + 1;
      *(_DWORD *)(v26 + 8) = v10;
      goto LABEL_9;
    }
    sub_2574E40((__int64)(v9 + 19), v27);
    sub_2567D60((__int64)(v9 + 19), (__int64 *)&v42, &v41);
    v26 = v41;
    v30 = *((_DWORD *)v9 + 42) + 1;
    goto LABEL_27;
  }
  v10 = *(unsigned int *)(v40 + 8);
LABEL_9:
  v9[23][2 * v10 + 1] = (__int64)v7;
  return 1;
}
