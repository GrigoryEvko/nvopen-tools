// Function: sub_27B1700
// Address: 0x27b1700
//
__int64 __fastcall sub_27B1700(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // r14
  __int64 i; // rax
  int v9; // ecx
  __int64 v10; // r10
  __int64 v11; // rbx
  __int64 *v12; // r12
  __int64 v13; // r8
  __int64 v14; // rdi
  int v15; // ecx
  unsigned int v16; // esi
  __int64 *v17; // rdx
  __int64 v18; // r15
  unsigned int v19; // r15d
  unsigned int v20; // esi
  __int64 *v21; // rdx
  __int64 v22; // r11
  _QWORD *v23; // rax
  __int64 v24; // rax
  __int64 v25; // r14
  unsigned int v26; // edi
  __int64 v27; // r10
  __int64 v28; // rdx
  int v29; // r11d
  _QWORD *v30; // rcx
  unsigned int v31; // r8d
  __int64 v32; // rsi
  __int64 v33; // r9
  unsigned int v34; // r8d
  __int64 v35; // r9
  _QWORD *v36; // rsi
  __int64 *v37; // rbx
  __int64 v38; // rax
  int v40; // esi
  int v41; // r9d
  __int64 v42; // rsi
  int v43; // esi
  int v44; // edx
  int v45; // edx
  __int64 v46; // rax
  __int64 *v47; // rax
  int v48; // r11d
  int v49; // [rsp+Ch] [rbp-74h]
  int v50; // [rsp+Ch] [rbp-74h]
  unsigned int v53; // [rsp+18h] [rbp-68h]
  int v54; // [rsp+18h] [rbp-68h]
  unsigned int v56; // [rsp+20h] [rbp-60h]
  __int64 v58; // [rsp+30h] [rbp-50h] BYREF
  __int64 v59; // [rsp+38h] [rbp-48h]
  __int64 v60; // [rsp+40h] [rbp-40h]
  __int64 v61; // [rsp+48h] [rbp-38h]

  v7 = (a3 - 1) / 2;
  if ( a2 >= v7 )
  {
    v11 = a2;
    v24 = a3;
    v12 = (__int64 *)(a1 + 16 * a2);
    if ( (a3 & 1) != 0 )
      goto LABEL_12;
    goto LABEL_37;
  }
  for ( i = a2; ; i = v11 )
  {
    v9 = *(_DWORD *)(a6 + 24);
    v10 = *(_QWORD *)(a6 + 8);
    v11 = 2 * (i + 1);
    v12 = (__int64 *)(a1 + 32 * (i + 1));
    v13 = *(v12 - 2);
    v14 = *v12;
    if ( v9 )
    {
      v15 = v9 - 1;
      v16 = v15 & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
      v17 = (__int64 *)(v10 + 16LL * v16);
      v18 = *v17;
      if ( v14 == *v17 )
      {
LABEL_6:
        v19 = *((_DWORD *)v17 + 2);
      }
      else
      {
        v45 = 1;
        while ( v18 != -4096 )
        {
          v48 = v45 + 1;
          v16 = v15 & (v45 + v16);
          v17 = (__int64 *)(v10 + 16LL * v16);
          v18 = *v17;
          if ( v14 == *v17 )
            goto LABEL_6;
          v45 = v48;
        }
        v19 = 0;
      }
      v20 = v15 & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
      v21 = (__int64 *)(v10 + 16LL * v20);
      v22 = *v21;
      if ( v13 == *v21 )
      {
LABEL_8:
        if ( *((_DWORD *)v21 + 2) > v19 )
        {
          --v11;
          v12 = (__int64 *)(a1 + 16 * v11);
          v14 = *v12;
        }
      }
      else
      {
        v44 = 1;
        while ( v22 != -4096 )
        {
          v20 = v15 & (v44 + v20);
          v50 = v44 + 1;
          v21 = (__int64 *)(v10 + 16LL * v20);
          v22 = *v21;
          if ( v13 == *v21 )
            goto LABEL_8;
          v44 = v50;
        }
      }
    }
    v23 = (_QWORD *)(a1 + 16 * i);
    *v23 = v14;
    v23[1] = v12[1];
    if ( v11 >= v7 )
      break;
  }
  v24 = a3;
  if ( (a3 & 1) == 0 )
  {
LABEL_37:
    if ( (v24 - 2) / 2 == v11 )
    {
      v46 = v11 + 1;
      v11 = 2 * (v11 + 1) - 1;
      v47 = (__int64 *)(a1 + 32 * v46 - 16);
      *v12 = *v47;
      v12[1] = v47[1];
      v12 = (__int64 *)(a1 + 16 * v11);
    }
  }
LABEL_12:
  v58 = 0;
  v59 = 0;
  v60 = 0;
  v61 = 0;
  sub_27B1670((__int64)&v58, a6);
  v25 = a2;
  v26 = v61;
  v27 = v59;
  v28 = (v11 - 1) / 2;
  if ( v11 > a2 )
  {
    v29 = v61 - 1;
    v56 = (v61 - 1) & (((unsigned int)a4 >> 9) ^ ((unsigned int)a4 >> 4));
    v30 = (_QWORD *)(v59 + 16LL * v56);
    while ( 1 )
    {
      v12 = (__int64 *)(a1 + 16 * v28);
      v38 = *v12;
      if ( !v26 )
        goto LABEL_21;
      v31 = v29 & (((unsigned int)v38 >> 9) ^ ((unsigned int)v38 >> 4));
      v32 = v27 + 16LL * v31;
      v33 = *(_QWORD *)v32;
      if ( v38 == *(_QWORD *)v32 )
      {
LABEL_15:
        v34 = *(_DWORD *)(v32 + 8);
      }
      else
      {
        v43 = 1;
        while ( v33 != -4096 )
        {
          v31 = v29 & (v43 + v31);
          v54 = v43 + 1;
          v32 = v27 + 16LL * v31;
          v33 = *(_QWORD *)v32;
          if ( v38 == *(_QWORD *)v32 )
            goto LABEL_15;
          v43 = v54;
        }
        v34 = 0;
      }
      v35 = *v30;
      v36 = v30;
      if ( *v30 != a4 )
      {
        v53 = v56;
        v40 = 1;
        while ( v35 != -4096 )
        {
          v41 = v40 + 1;
          v42 = v29 & (v53 + v40);
          v49 = v41;
          v53 = v42;
          v36 = (_QWORD *)(v27 + 16 * v42);
          v35 = *v36;
          if ( a4 == *v36 )
            goto LABEL_17;
          v40 = v49;
        }
LABEL_21:
        v12 = (__int64 *)(a1 + 16 * v11);
        goto LABEL_22;
      }
LABEL_17:
      v37 = (__int64 *)(a1 + 16 * v11);
      if ( *((_DWORD *)v36 + 2) <= v34 )
        break;
      *v37 = v38;
      v37[1] = v12[1];
      v11 = v28;
      if ( v25 >= v28 )
        goto LABEL_22;
      v28 = (v28 - 1) / 2;
    }
    v12 = v37;
  }
LABEL_22:
  *v12 = a4;
  v12[1] = a5;
  return sub_C7D6A0(v27, 16LL * v26, 8);
}
