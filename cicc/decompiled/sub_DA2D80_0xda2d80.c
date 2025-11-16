// Function: sub_DA2D80
// Address: 0xda2d80
//
__int64 __fastcall sub_DA2D80(__int64 a1, _QWORD *a2, _BYTE *a3, __int64 a4, __int64 a5, unsigned int a6)
{
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rsi
  char v15; // r10
  __int64 v16; // rdi
  int v17; // r9d
  unsigned int v18; // r8d
  __int64 v19; // r11
  __int64 v20; // rax
  char v21; // r11
  __int64 v22; // rax
  __int64 v23; // rdx
  __int64 v24; // rcx
  __int64 v25; // r8
  __int64 v26; // r9
  __int64 v28; // r8
  __int64 v29; // rax
  __int64 v30; // rax
  _BYTE *v31; // rax
  char v32; // dl
  __int64 v33; // r15
  __int64 v34; // rax
  int v35; // r8d
  unsigned int v36; // r10d
  __int64 v37; // rdi
  __int64 v38; // rdx
  bool v39; // al
  unsigned __int64 v40; // r14
  __int64 v41; // rdi
  __int64 v42; // rax
  __int64 v43; // rax
  __int64 v44; // r11
  unsigned __int64 v45; // r9
  int v46; // eax
  __int64 v47; // r9
  __int64 v48; // rdi
  __int64 v49; // rdx
  __int64 v50; // rdx
  _BYTE *v51; // rsi
  __int64 v52; // rax
  __int64 v53; // rdx
  __int64 v54; // rcx
  __int64 v55; // r8
  unsigned int v56; // r14d
  __int64 v57; // rax
  _QWORD *v58; // r14
  __int64 v59; // rax
  __int64 v60; // r9
  int v61; // eax
  bool v62; // al
  __int64 v63; // rax
  __int64 v64; // rdx
  __int64 v65; // rcx
  __int64 v66; // r8
  __int64 v67; // r9
  int v68; // eax
  int v69; // [rsp+8h] [rbp-78h]
  char v70; // [rsp+Ch] [rbp-74h]
  int v71; // [rsp+Ch] [rbp-74h]
  int v72; // [rsp+10h] [rbp-70h]
  __int64 v73; // [rsp+18h] [rbp-68h]
  char v74; // [rsp+18h] [rbp-68h]
  __int64 v75; // [rsp+20h] [rbp-60h]
  _BYTE *v76; // [rsp+20h] [rbp-60h]
  int v77; // [rsp+28h] [rbp-58h]
  int v78; // [rsp+28h] [rbp-58h]
  __int64 v80; // [rsp+30h] [rbp-50h] BYREF
  unsigned int v81; // [rsp+38h] [rbp-48h]
  __int64 v82; // [rsp+40h] [rbp-40h] BYREF
  unsigned int v83; // [rsp+48h] [rbp-38h]

  if ( *(_BYTE *)a4 != 17 )
    goto LABEL_14;
  v11 = sub_D47930(a5);
  if ( !v11 )
    goto LABEL_14;
  v75 = v11;
  v12 = sub_D47840(a5);
  v13 = v75;
  v14 = v12;
  if ( !v12 )
    goto LABEL_14;
  v15 = *a3;
  if ( *a3 == 55 )
  {
    v76 = (_BYTE *)*((_QWORD *)a3 - 8);
    if ( !v76 )
      goto LABEL_12;
    v16 = *((_QWORD *)a3 - 4);
    v17 = 26;
    if ( *(_BYTE *)v16 != 17 )
      goto LABEL_12;
  }
  else if ( v15 == 56 )
  {
    v76 = (_BYTE *)*((_QWORD *)a3 - 8);
    if ( !v76 )
      goto LABEL_12;
    v16 = *((_QWORD *)a3 - 4);
    v17 = 27;
    if ( *(_BYTE *)v16 != 17 )
      goto LABEL_12;
  }
  else
  {
    if ( v15 != 54 )
      goto LABEL_12;
    v76 = (_BYTE *)*((_QWORD *)a3 - 8);
    if ( !v76 )
      goto LABEL_14;
    v16 = *((_QWORD *)a3 - 4);
    if ( *(_BYTE *)v16 != 17 )
      goto LABEL_14;
    v17 = 25;
  }
  v18 = *(_DWORD *)(v16 + 32);
  v19 = 1LL << ((unsigned __int8)v18 - 1);
  v20 = *(_QWORD *)(v16 + 24);
  if ( v18 > 0x40 )
  {
    if ( (*(_QWORD *)(v20 + 8LL * ((v18 - 1) >> 6)) & v19) != 0 )
      goto LABEL_12;
    v69 = *(_DWORD *)(v16 + 32);
    v70 = *a3;
    v77 = v17;
    v73 = v13;
    v61 = sub_C444A0(v16 + 24);
    v13 = v73;
    v17 = v77;
    v15 = v70;
    v62 = v69 == v61;
  }
  else
  {
    if ( (v20 & v19) != 0 )
      goto LABEL_12;
    v62 = v20 == 0;
  }
  if ( !v62 )
  {
    v21 = 1;
    v15 = *v76;
    a3 = v76;
    goto LABEL_13;
  }
LABEL_12:
  v21 = 0;
  v17 = 0;
LABEL_13:
  if ( v15 != 84 || *((_QWORD *)a3 + 5) != **(_QWORD **)(a5 + 32) )
    goto LABEL_14;
  v28 = *((_QWORD *)a3 - 1);
  v29 = 0x1FFFFFFFE0LL;
  if ( (*((_DWORD *)a3 + 1) & 0x7FFFFFF) != 0 )
  {
    v30 = 0;
    do
    {
      if ( v13 == *(_QWORD *)(v28 + 32LL * *((unsigned int *)a3 + 18) + 8 * v30) )
      {
        v29 = 32 * v30;
        goto LABEL_25;
      }
      ++v30;
    }
    while ( (*((_DWORD *)a3 + 1) & 0x7FFFFFF) != (_DWORD)v30 );
    v29 = 0x1FFFFFFFE0LL;
  }
LABEL_25:
  v31 = *(_BYTE **)(v28 + v29);
  v32 = *v31;
  if ( *v31 == 55 )
  {
    v33 = *((_QWORD *)v31 - 8);
    if ( !v33 )
      goto LABEL_14;
    v34 = *((_QWORD *)v31 - 4);
    v35 = 26;
    if ( *(_BYTE *)v34 != 17 )
      goto LABEL_14;
  }
  else if ( v32 == 56 )
  {
    v33 = *((_QWORD *)v31 - 8);
    if ( !v33 )
      goto LABEL_14;
    v34 = *((_QWORD *)v31 - 4);
    v35 = 27;
    if ( *(_BYTE *)v34 != 17 )
      goto LABEL_14;
  }
  else
  {
    if ( v32 != 54 )
      goto LABEL_14;
    v33 = *((_QWORD *)v31 - 8);
    if ( !v33 )
      goto LABEL_14;
    v34 = *((_QWORD *)v31 - 4);
    if ( *(_BYTE *)v34 != 17 )
      goto LABEL_14;
    v35 = 25;
  }
  v36 = *(_DWORD *)(v34 + 32);
  v37 = *(_QWORD *)(v34 + 24);
  v38 = 1LL << ((unsigned __int8)v36 - 1);
  if ( v36 > 0x40 )
  {
    if ( (*(_QWORD *)(v37 + 8LL * ((v36 - 1) >> 6)) & v38) != 0 )
      goto LABEL_14;
    v71 = v35;
    v78 = *(_DWORD *)(v34 + 32);
    v72 = v17;
    v74 = v21;
    v68 = sub_C444A0(v34 + 24);
    v21 = v74;
    v17 = v72;
    v35 = v71;
    v39 = v78 == v68;
  }
  else
  {
    if ( (v38 & v37) != 0 )
      goto LABEL_14;
    v39 = v37 == 0;
  }
  if ( !v39 && a3 == (_BYTE *)v33 && (!v21 || v17 == v35) )
  {
    v40 = a2[1];
    if ( v35 == 27 )
    {
      v41 = *(_QWORD *)(v33 - 8);
      v42 = 0x1FFFFFFFE0LL;
      if ( (*(_DWORD *)(v33 + 4) & 0x7FFFFFF) != 0 )
      {
        v43 = 0;
        do
        {
          if ( v14 == *(_QWORD *)(v41 + 32LL * *(unsigned int *)(v33 + 72) + 8 * v43) )
          {
            v42 = 32 * v43;
            goto LABEL_44;
          }
          ++v43;
        }
        while ( (*(_DWORD *)(v33 + 4) & 0x7FFFFFF) != (_DWORD)v43 );
        v42 = 0x1FFFFFFFE0LL;
      }
LABEL_44:
      v44 = *(_QWORD *)(v41 + v42);
      v45 = *(_QWORD *)(v14 + 48) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v45 == v14 + 48 )
      {
        v47 = 0;
      }
      else
      {
        if ( !v45 )
          BUG();
        v46 = *(unsigned __int8 *)(v45 - 24);
        v47 = v45 - 24;
        if ( (unsigned int)(v46 - 30) >= 0xB )
          v47 = 0;
      }
      sub_9AC3E0((__int64)&v80, v44, v40, 0, a2[4], v47, a2[5], 1);
      v48 = *(_QWORD *)(a4 + 8);
      v49 = v81 > 0x40 ? *(_QWORD *)(v80 + 8LL * ((v81 - 1) >> 6)) : v80;
      if ( (v49 & (1LL << ((unsigned __int8)v81 - 1))) != 0 )
      {
        v51 = (_BYTE *)sub_ACD640(v48, 0, 0);
      }
      else
      {
        v50 = v82;
        if ( v83 > 0x40 )
          v50 = *(_QWORD *)(v82 + 8LL * ((v83 - 1) >> 6));
        if ( (v50 & (1LL << ((unsigned __int8)v83 - 1))) == 0 )
        {
          v63 = sub_D970F0((__int64)a2);
          sub_D97F80(a1, v63, v64, v65, v66, v67);
          sub_969240(&v82);
          sub_969240(&v80);
          return a1;
        }
        v51 = (_BYTE *)sub_ACD640(v48, -1, 1u);
      }
      if ( v83 > 0x40 && v82 )
        j_j___libc_free_0_0(v82);
      if ( v81 > 0x40 && v80 )
        j_j___libc_free_0_0(v80);
    }
    else
    {
      v51 = (_BYTE *)sub_ACD640(*(_QWORD *)(a4 + 8), 0, 0);
    }
    v52 = sub_9719A0(a6, v51, a4, v40, a2[3], 0);
    if ( sub_AD7890(v52, (__int64)v51, v53, v54, v55) )
    {
      v56 = sub_D97050((__int64)a2, *(_QWORD *)(a4 + 8));
      v57 = sub_D97090((__int64)a2, *(_QWORD *)(a4 + 8));
      v58 = sub_DA2C50((__int64)a2, v57, v56, 0);
      v59 = sub_D970F0((__int64)a2);
      sub_D97D90(a1, v59, (__int64)v58, (__int64)v58, 0, v60, 0, 0);
      return a1;
    }
  }
LABEL_14:
  v22 = sub_D970F0((__int64)a2);
  sub_D97F80(a1, v22, v23, v24, v25, v26);
  return a1;
}
