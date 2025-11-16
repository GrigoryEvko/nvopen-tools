// Function: sub_28D7150
// Address: 0x28d7150
//
__int64 __fastcall sub_28D7150(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v7; // r14
  unsigned int v8; // r9d
  __int64 v9; // r8
  unsigned int v10; // edi
  __int64 *v11; // rax
  __int64 v12; // r11
  unsigned int v13; // r11d
  unsigned int v14; // ecx
  __int64 *v15; // rax
  __int64 v16; // r10
  __int64 v17; // r12
  __int64 *v18; // r13
  unsigned int v19; // esi
  __int64 v20; // rdx
  __int64 v21; // rcx
  __int64 *v22; // r10
  int v23; // edx
  __int64 v24; // r14
  __int64 v25; // r11
  unsigned int v26; // r8d
  __int64 v27; // rdi
  unsigned int v28; // ecx
  __int64 *v29; // rax
  __int64 v30; // r10
  unsigned int v31; // r9d
  __int64 v32; // rcx
  unsigned int v33; // edx
  __int64 *v34; // rax
  __int64 v35; // r10
  __int64 *v36; // rdx
  unsigned int v37; // esi
  __int64 v38; // rdx
  __int64 *v39; // r9
  int v40; // ecx
  __int64 v41; // r12
  __int64 v42; // rax
  __int64 *v43; // r9
  int v44; // eax
  int v45; // eax
  int v47; // eax
  __int64 *v48; // rdi
  int v49; // eax
  int v50; // eax
  int v51; // eax
  int v52; // [rsp+14h] [rbp-8Ch]
  int v53; // [rsp+14h] [rbp-8Ch]
  __int64 v55; // [rsp+20h] [rbp-80h]
  unsigned int v56; // [rsp+20h] [rbp-80h]
  int v57; // [rsp+28h] [rbp-78h]
  int v58; // [rsp+28h] [rbp-78h]
  __int64 v60; // [rsp+38h] [rbp-68h]
  __int64 v61; // [rsp+38h] [rbp-68h]
  __int64 *v62; // [rsp+38h] [rbp-68h]
  __int64 v63; // [rsp+40h] [rbp-60h]
  __int64 v64; // [rsp+40h] [rbp-60h]
  __int64 v66; // [rsp+58h] [rbp-48h] BYREF
  __int64 v67; // [rsp+60h] [rbp-40h] BYREF
  _QWORD v68[7]; // [rsp+68h] [rbp-38h] BYREF

  v55 = a3 & 1;
  v63 = (a3 - 1) / 2;
  if ( a2 < v63 )
  {
    v7 = a2;
    v60 = a5 + 1360;
    while ( 1 )
    {
      v19 = *(_DWORD *)(a5 + 1384);
      v17 = 2 * (v7 + 1);
      v18 = (__int64 *)(a1 + 16 * (v7 + 1));
      v20 = *(v18 - 1);
      v21 = *v18;
      v67 = v20;
      v66 = v21;
      if ( v19 )
      {
        v8 = v19 - 1;
        v9 = *(_QWORD *)(a5 + 1368);
        v10 = (v19 - 1) & (((unsigned int)v21 >> 9) ^ ((unsigned int)v21 >> 4));
        v11 = (__int64 *)(v9 + 16LL * v10);
        v12 = *v11;
        if ( v21 == *v11 )
        {
LABEL_4:
          v13 = *((_DWORD *)v11 + 2);
          goto LABEL_5;
        }
        v53 = 1;
        v22 = 0;
        while ( v12 != -4096 )
        {
          if ( v12 == -8192 && !v22 )
            v22 = v11;
          v10 = v8 & (v53 + v10);
          v11 = (__int64 *)(v9 + 16LL * v10);
          v12 = *v11;
          if ( v21 == *v11 )
            goto LABEL_4;
          ++v53;
        }
        if ( !v22 )
          v22 = v11;
        v51 = *(_DWORD *)(a5 + 1376);
        ++*(_QWORD *)(a5 + 1360);
        v23 = v51 + 1;
        v68[0] = v22;
        if ( 4 * (v51 + 1) < 3 * v19 )
        {
          if ( v19 - *(_DWORD *)(a5 + 1380) - v23 > v19 >> 3 )
            goto LABEL_68;
          goto LABEL_13;
        }
      }
      else
      {
        ++*(_QWORD *)(a5 + 1360);
        v68[0] = 0;
      }
      v19 *= 2;
LABEL_13:
      sub_CE3370(v60, v19);
      sub_28CD4F0(v60, &v66, v68);
      v21 = v66;
      v22 = (__int64 *)v68[0];
      v23 = *(_DWORD *)(a5 + 1376) + 1;
LABEL_68:
      *(_DWORD *)(a5 + 1376) = v23;
      if ( *v22 != -4096 )
        --*(_DWORD *)(a5 + 1380);
      *v22 = v21;
      *((_DWORD *)v22 + 2) = 0;
      v19 = *(_DWORD *)(a5 + 1384);
      if ( !v19 )
      {
        ++*(_QWORD *)(a5 + 1360);
        v68[0] = 0;
        goto LABEL_72;
      }
      v9 = *(_QWORD *)(a5 + 1368);
      v20 = v67;
      v8 = v19 - 1;
      v13 = 0;
LABEL_5:
      v14 = v8 & (((unsigned int)v20 >> 9) ^ ((unsigned int)v20 >> 4));
      v15 = (__int64 *)(v9 + 16LL * v14);
      v16 = *v15;
      if ( *v15 != v20 )
      {
        v52 = 1;
        v48 = 0;
        while ( v16 != -4096 )
        {
          if ( !v48 && v16 == -8192 )
            v48 = v15;
          v14 = v8 & (v52 + v14);
          v15 = (__int64 *)(v9 + 16LL * v14);
          v16 = *v15;
          if ( *v15 == v20 )
            goto LABEL_6;
          ++v52;
        }
        if ( !v48 )
          v48 = v15;
        v49 = *(_DWORD *)(a5 + 1376);
        ++*(_QWORD *)(a5 + 1360);
        v50 = v49 + 1;
        v68[0] = v48;
        if ( 4 * v50 >= 3 * v19 )
        {
LABEL_72:
          v19 *= 2;
        }
        else if ( v19 - (v50 + *(_DWORD *)(a5 + 1380)) > v19 >> 3 )
        {
LABEL_59:
          *(_DWORD *)(a5 + 1376) = v50;
          if ( *v48 != -4096 )
            --*(_DWORD *)(a5 + 1380);
          *v48 = v20;
          *((_DWORD *)v48 + 2) = 0;
          goto LABEL_8;
        }
        sub_CE3370(v60, v19);
        sub_28CD4F0(v60, &v67, v68);
        v20 = v67;
        v48 = (__int64 *)v68[0];
        v50 = *(_DWORD *)(a5 + 1376) + 1;
        goto LABEL_59;
      }
LABEL_6:
      if ( v13 < *((_DWORD *)v15 + 2) )
      {
        --v17;
        v18 = (__int64 *)(a1 + 8 * v17);
      }
LABEL_8:
      *(_QWORD *)(a1 + 8 * v7) = *v18;
      if ( v17 >= v63 )
      {
        if ( v55 )
          goto LABEL_15;
        goto LABEL_28;
      }
      v7 = v17;
    }
  }
  v17 = a2;
  v18 = (__int64 *)(a1 + 8 * a2);
  if ( (a3 & 1) != 0 )
    goto LABEL_39;
LABEL_28:
  if ( (a3 - 2) / 2 == v17 )
  {
    v41 = 2 * v17 + 2;
    v42 = *(_QWORD *)(a1 + 8 * v41 - 8);
    v17 = v41 - 1;
    *v18 = v42;
    v18 = (__int64 *)(a1 + 8 * v17);
  }
LABEL_15:
  v24 = (v17 - 1) / 2;
  if ( v17 <= a2 )
    goto LABEL_39;
  v25 = a4;
  v64 = a5 + 1360;
  while ( 1 )
  {
    v18 = (__int64 *)(a1 + 8 * v24);
    v37 = *(_DWORD *)(a5 + 1384);
    v67 = v25;
    v38 = *v18;
    v66 = *v18;
    if ( !v37 )
    {
      ++*(_QWORD *)(a5 + 1360);
      v68[0] = 0;
LABEL_25:
      v61 = v25;
      v37 *= 2;
      goto LABEL_26;
    }
    v26 = v37 - 1;
    v27 = *(_QWORD *)(a5 + 1368);
    v28 = (v37 - 1) & (((unsigned int)v38 >> 9) ^ ((unsigned int)v38 >> 4));
    v29 = (__int64 *)(v27 + 16LL * v28);
    v30 = *v29;
    if ( *v29 == v38 )
    {
LABEL_18:
      v31 = *((_DWORD *)v29 + 2);
      v32 = v25;
      goto LABEL_19;
    }
    v58 = 1;
    v39 = 0;
    while ( v30 != -4096 )
    {
      if ( v30 == -8192 && !v39 )
        v39 = v29;
      v28 = v26 & (v58 + v28);
      v29 = (__int64 *)(v27 + 16LL * v28);
      v30 = *v29;
      if ( v38 == *v29 )
        goto LABEL_18;
      ++v58;
    }
    if ( !v39 )
      v39 = v29;
    v47 = *(_DWORD *)(a5 + 1376);
    ++*(_QWORD *)(a5 + 1360);
    v40 = v47 + 1;
    v68[0] = v39;
    if ( 4 * (v47 + 1) >= 3 * v37 )
      goto LABEL_25;
    if ( v37 - *(_DWORD *)(a5 + 1380) - v40 > v37 >> 3 )
      goto LABEL_46;
    v61 = v25;
LABEL_26:
    sub_CE3370(v64, v37);
    sub_28CD4F0(v64, &v66, v68);
    v38 = v66;
    v39 = (__int64 *)v68[0];
    v25 = v61;
    v40 = *(_DWORD *)(a5 + 1376) + 1;
LABEL_46:
    *(_DWORD *)(a5 + 1376) = v40;
    if ( *v39 != -4096 )
      --*(_DWORD *)(a5 + 1380);
    *v39 = v38;
    *((_DWORD *)v39 + 2) = 0;
    v37 = *(_DWORD *)(a5 + 1384);
    if ( !v37 )
    {
      ++*(_QWORD *)(a5 + 1360);
      v68[0] = 0;
      goto LABEL_50;
    }
    v27 = *(_QWORD *)(a5 + 1368);
    v32 = v67;
    v26 = v37 - 1;
    v31 = 0;
LABEL_19:
    v33 = v26 & (((unsigned int)v32 >> 9) ^ ((unsigned int)v32 >> 4));
    v34 = (__int64 *)(v27 + 16LL * v33);
    v35 = *v34;
    if ( *v34 != v32 )
      break;
LABEL_20:
    v36 = (__int64 *)(a1 + 8 * v17);
    if ( v31 >= *((_DWORD *)v34 + 2) )
    {
      v18 = (__int64 *)(a1 + 8 * v17);
      goto LABEL_39;
    }
    v17 = v24;
    *v36 = *v18;
    if ( a2 >= v24 )
      goto LABEL_39;
    v24 = (v24 - 1) / 2;
  }
  v57 = 1;
  v62 = 0;
  v56 = v31;
  while ( 1 )
  {
    v43 = v62;
    if ( v35 == -4096 )
      break;
    if ( !v62 )
    {
      if ( v35 != -8192 )
        v34 = 0;
      v62 = v34;
    }
    v33 = v26 & (v57 + v33);
    v34 = (__int64 *)(v27 + 16LL * v33);
    v35 = *v34;
    if ( *v34 == v32 )
    {
      v31 = v56;
      goto LABEL_20;
    }
    ++v57;
  }
  if ( !v62 )
    v43 = v34;
  v44 = *(_DWORD *)(a5 + 1376);
  ++*(_QWORD *)(a5 + 1360);
  v45 = v44 + 1;
  v68[0] = v43;
  if ( 4 * v45 < 3 * v37 )
  {
    if ( v37 - (v45 + *(_DWORD *)(a5 + 1380)) <= v37 >> 3 )
      goto LABEL_51;
    goto LABEL_36;
  }
LABEL_50:
  v37 *= 2;
LABEL_51:
  sub_CE3370(v64, v37);
  sub_28CD4F0(v64, &v67, v68);
  v32 = v67;
  v43 = (__int64 *)v68[0];
  v45 = *(_DWORD *)(a5 + 1376) + 1;
LABEL_36:
  *(_DWORD *)(a5 + 1376) = v45;
  if ( *v43 != -4096 )
    --*(_DWORD *)(a5 + 1380);
  *v43 = v32;
  v18 = (__int64 *)(a1 + 8 * v17);
  *((_DWORD *)v43 + 2) = 0;
LABEL_39:
  *v18 = a4;
  return a4;
}
