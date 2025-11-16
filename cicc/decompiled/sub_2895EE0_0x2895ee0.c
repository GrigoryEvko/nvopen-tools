// Function: sub_2895EE0
// Address: 0x2895ee0
//
__int64 __fastcall sub_2895EE0(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r14
  int v10; // eax
  _QWORD *v11; // rdi
  _QWORD *v12; // rsi
  __int64 v13; // rdx
  __int64 v14; // r8
  int v16; // eax
  __int64 v17; // rdi
  unsigned int v18; // eax
  __int64 v19; // r15
  _QWORD *v20; // rax
  __int64 v21; // rsi
  char v22; // dl
  unsigned int v23; // esi
  __int64 v24; // r8
  unsigned int v25; // eax
  __int64 *v26; // rdi
  __int64 v27; // rdx
  __int64 v28; // r8
  __int64 v29; // rax
  __int64 v30; // r9
  unsigned int v31; // esi
  __int64 *v32; // rdx
  __int64 v33; // r10
  __int64 v34; // rsi
  _DWORD *v35; // rax
  int v36; // r11d
  int v37; // r10d
  __int64 v38; // rdx
  __int64 *v39; // rax
  __int64 *v40; // r15
  __int64 v41; // r9
  int v42; // r12d
  int v43; // r13d
  __int64 v44; // rdx
  int v45; // r8d
  int v46; // edi
  int v47; // ecx
  int v48; // edx
  int v49; // ecx
  __int64 v50; // [rsp+8h] [rbp-98h]
  int v51; // [rsp+18h] [rbp-88h]
  int v52; // [rsp+1Ch] [rbp-84h]
  int v53; // [rsp+20h] [rbp-80h]
  int v54; // [rsp+24h] [rbp-7Ch]
  __int64 *v55; // [rsp+28h] [rbp-78h]
  int v56; // [rsp+30h] [rbp-70h]
  int v57; // [rsp+34h] [rbp-6Ch]
  __int64 v58; // [rsp+38h] [rbp-68h]
  __int64 v60; // [rsp+48h] [rbp-58h] BYREF
  _DWORD v61[20]; // [rsp+50h] [rbp-50h] BYREF

  v6 = a4;
  v10 = *(_DWORD *)(a5 + 16);
  v60 = a3;
  if ( v10 )
  {
    v16 = *(_DWORD *)(a5 + 24);
    v17 = *(_QWORD *)(a5 + 8);
    if ( !v16 )
      goto LABEL_3;
    v13 = (unsigned int)(v16 - 1);
    v18 = v13 & (((unsigned int)v60 >> 9) ^ ((unsigned int)v60 >> 4));
    v14 = v18;
    v19 = *(_QWORD *)(v17 + 8LL * v18);
    if ( v60 != v19 )
    {
      v45 = 1;
      while ( v19 != -4096 )
      {
        a4 = (unsigned int)(v45 + 1);
        v18 = v13 & (v45 + v18);
        v14 = v18;
        v19 = *(_QWORD *)(v17 + 8LL * v18);
        if ( v60 == v19 )
          goto LABEL_7;
        v45 = a4;
      }
      goto LABEL_3;
    }
LABEL_7:
    if ( !*(_BYTE *)(v6 + 28) )
      goto LABEL_14;
  }
  else
  {
    v11 = *(_QWORD **)(a5 + 32);
    v12 = &v11[*(unsigned int *)(a5 + 40)];
    if ( v12 == sub_28946A0(v11, (__int64)v12, &v60) )
    {
LABEL_3:
      *(_QWORD *)a1 = 0;
      *(_QWORD *)(a1 + 8) = 0;
      *(_QWORD *)(a1 + 16) = 0;
      *(_QWORD *)(a1 + 24) = 0;
      return a1;
    }
    v19 = v60;
    if ( !*(_BYTE *)(v6 + 28) )
      goto LABEL_14;
  }
  v20 = *(_QWORD **)(v6 + 8);
  v21 = *(unsigned int *)(v6 + 20);
  v13 = (__int64)&v20[v21];
  if ( v20 != (_QWORD *)v13 )
  {
    while ( v19 != *v20 )
    {
      if ( (_QWORD *)v13 == ++v20 )
        goto LABEL_30;
    }
    goto LABEL_3;
  }
LABEL_30:
  if ( (unsigned int)v21 < *(_DWORD *)(v6 + 16) )
  {
    *(_DWORD *)(v6 + 20) = v21 + 1;
    *(_QWORD *)v13 = v19;
    ++*(_QWORD *)v6;
    goto LABEL_15;
  }
LABEL_14:
  sub_C8CC70(v6, v19, v13, a4, v14, a6);
  if ( !v22 )
    goto LABEL_3;
LABEL_15:
  v23 = *(_DWORD *)(a6 + 24);
  v24 = *(_QWORD *)(a6 + 8);
  if ( v23 )
  {
    v25 = (v23 - 1) & (((unsigned int)v19 >> 9) ^ ((unsigned int)v19 >> 4));
    v26 = (__int64 *)(v24 + 56LL * v25);
    v27 = *v26;
    if ( v19 == *v26 )
      goto LABEL_17;
    v46 = 1;
    while ( v27 != -4096 )
    {
      v47 = v46 + 1;
      v25 = (v23 - 1) & (v25 + v46);
      v26 = (__int64 *)(v24 + 56LL * v25);
      v27 = *v26;
      if ( v19 == *v26 )
        goto LABEL_17;
      v46 = v47;
    }
  }
  v26 = (__int64 *)(v24 + 56LL * v23);
LABEL_17:
  v28 = *a2;
  v29 = *(unsigned int *)(*a2 + 24);
  v30 = *(_QWORD *)(*a2 + 8);
  if ( !(_DWORD)v29 )
  {
LABEL_38:
    v34 = *(_QWORD *)(v28 + 32);
    goto LABEL_39;
  }
  v31 = (v29 - 1) & (((unsigned int)v19 >> 9) ^ ((unsigned int)v19 >> 4));
  v32 = (__int64 *)(v30 + 16LL * v31);
  v33 = *v32;
  if ( v19 != *v32 )
  {
    v48 = 1;
    while ( v33 != -4096 )
    {
      v49 = v48 + 1;
      v31 = (v29 - 1) & (v48 + v31);
      v32 = (__int64 *)(v30 + 16LL * v31);
      v33 = *v32;
      if ( v19 == *v32 )
        goto LABEL_19;
      v48 = v49;
    }
    goto LABEL_38;
  }
LABEL_19:
  v34 = *(_QWORD *)(v28 + 32);
  if ( v32 == (__int64 *)(v30 + 16 * v29) )
  {
LABEL_39:
    v35 = (_DWORD *)(v34 + 176LL * *(unsigned int *)(v28 + 40));
    goto LABEL_21;
  }
  v35 = (_DWORD *)(v34 + 176LL * *((unsigned int *)v32 + 2));
LABEL_21:
  v36 = v35[41];
  v37 = v35[40];
  v56 = v35[39];
  v57 = v35[38];
  if ( *((_DWORD *)v26 + 7) - *((_DWORD *)v26 + 8) == 1 )
  {
    v53 = v35[41];
    v36 = 0;
    v54 = v35[40];
    v37 = 0;
    v51 = v35[39];
    v52 = v35[38];
    v56 = 0;
    v57 = 0;
  }
  else
  {
    v53 = 0;
    v54 = 0;
    v51 = 0;
    v52 = 0;
  }
  v38 = 4LL * (*(_DWORD *)(v19 + 4) & 0x7FFFFFF);
  if ( (*(_BYTE *)(v19 + 7) & 0x40) != 0 )
  {
    v39 = *(__int64 **)(v19 - 8);
    v55 = &v39[v38];
  }
  else
  {
    v55 = (__int64 *)v19;
    v39 = (__int64 *)(v19 - v38 * 8);
  }
  if ( v55 != v39 )
  {
    v40 = v39;
    v50 = a1;
    v41 = a6;
    v42 = v37;
    v43 = v36;
    do
    {
      v44 = *v40;
      v58 = v41;
      v40 += 4;
      sub_2895EE0(v61, a2, v44, v6, a5);
      v42 += v61[6];
      v52 += v61[0];
      v43 += v61[7];
      v51 += v61[1];
      v54 += v61[2];
      v53 += v61[3];
      v57 += v61[4];
      v56 += v61[5];
      v41 = v58;
    }
    while ( v55 != v40 );
    v37 = v42;
    a1 = v50;
    v36 = v43;
  }
  *(_DWORD *)(a1 + 24) = v37;
  *(_DWORD *)(a1 + 28) = v36;
  *(_DWORD *)a1 = v52;
  *(_DWORD *)(a1 + 4) = v51;
  *(_DWORD *)(a1 + 8) = v54;
  *(_DWORD *)(a1 + 12) = v53;
  *(_DWORD *)(a1 + 16) = v57;
  *(_DWORD *)(a1 + 20) = v56;
  return a1;
}
