// Function: sub_DA6F50
// Address: 0xda6f50
//
_BYTE *__fastcall sub_DA6F50(__int64 a1, __int64 a2, __int64 a3, unsigned int a4)
{
  __int64 v6; // rax
  unsigned __int8 **v7; // r12
  _BYTE *v8; // r14
  _BYTE *v9; // rbx
  unsigned __int8 v10; // cl
  __int64 v11; // rsi
  _QWORD *v12; // rax
  _QWORD *v13; // rdx
  int v15; // ecx
  __int64 v16; // rsi
  int v17; // ecx
  unsigned int v18; // edx
  _QWORD *v19; // rax
  _BYTE *v20; // r8
  __int64 v21; // rax
  unsigned int v22; // esi
  __int64 v23; // r9
  unsigned int v24; // r8d
  _QWORD *v25; // rcx
  _BYTE *v26; // rdi
  __int64 *v27; // rdx
  int v28; // eax
  int v29; // edi
  _QWORD *v30; // rdx
  int v31; // edi
  int v32; // ecx
  int v33; // edx
  int v34; // r9d
  __int64 v35; // r11
  unsigned int v36; // r10d
  __int64 v37; // r8
  int v38; // edi
  _QWORD *v39; // rsi
  int v40; // edx
  int v41; // r9d
  int v42; // edi
  __int64 v43; // r10
  _BYTE *v44; // r8
  __int64 v45; // [rsp+8h] [rbp-58h]
  int v46; // [rsp+18h] [rbp-48h]
  __int64 v47; // [rsp+18h] [rbp-48h]
  __int64 v48; // [rsp+18h] [rbp-48h]
  unsigned int v49; // [rsp+24h] [rbp-3Ch]
  unsigned __int8 **v50; // [rsp+28h] [rbp-38h]

  if ( a4 > (unsigned int)qword_4F89228 )
    return 0;
  v6 = 4LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF);
  if ( (*(_BYTE *)(a1 + 7) & 0x40) != 0 )
  {
    v7 = *(unsigned __int8 ***)(a1 - 8);
    v50 = &v7[v6];
  }
  else
  {
    v50 = (unsigned __int8 **)a1;
    v7 = (unsigned __int8 **)(a1 - v6 * 8);
  }
  v8 = 0;
  if ( v7 != v50 )
  {
    v49 = a4 + 1;
    while ( 1 )
    {
      v9 = *v7;
      v10 = **v7;
      if ( v10 <= 0x15u )
        goto LABEL_18;
      if ( v10 <= 0x1Cu )
        return 0;
      v11 = *((_QWORD *)v9 + 5);
      if ( *(_BYTE *)(a2 + 84) )
      {
        v12 = *(_QWORD **)(a2 + 64);
        v13 = &v12[*(unsigned int *)(a2 + 76)];
        if ( v12 == v13 )
          return 0;
        while ( v11 != *v12 )
        {
          if ( v13 == ++v12 )
            return 0;
        }
        if ( v10 == 84 )
          goto LABEL_14;
      }
      else
      {
        if ( !sub_C8CA60(a2 + 56, v11) )
          return 0;
        if ( *v9 == 84 )
        {
LABEL_14:
          if ( **(_QWORD **)(a2 + 32) != *((_QWORD *)v9 + 5) )
            return 0;
          goto LABEL_15;
        }
      }
      if ( !sub_D90BC0(v9) )
        return 0;
      if ( *v9 == 84 )
        goto LABEL_15;
      v15 = *(_DWORD *)(a3 + 24);
      v16 = *(_QWORD *)(a3 + 8);
      if ( !v15 )
        goto LABEL_30;
      v17 = v15 - 1;
      v18 = v17 & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
      v19 = (_QWORD *)(v16 + 16LL * v18);
      v20 = (_BYTE *)*v19;
      if ( v9 != (_BYTE *)*v19 )
        break;
LABEL_29:
      v21 = v19[1];
      if ( !v21 )
        goto LABEL_30;
LABEL_34:
      v9 = (_BYTE *)v21;
LABEL_15:
      if ( v8 && v9 != v8 )
        return 0;
      v8 = v9;
LABEL_18:
      v7 += 4;
      if ( v50 == v7 )
        return v8;
    }
    v28 = 1;
    while ( v20 != (_BYTE *)-4096LL )
    {
      v29 = v28 + 1;
      v18 = v17 & (v28 + v18);
      v19 = (_QWORD *)(v16 + 16LL * v18);
      v20 = (_BYTE *)*v19;
      if ( v9 == (_BYTE *)*v19 )
        goto LABEL_29;
      v28 = v29;
    }
LABEL_30:
    v21 = sub_DA6F50(v9, a2, a3, v49);
    v22 = *(_DWORD *)(a3 + 24);
    if ( v22 )
    {
      v23 = *(_QWORD *)(a3 + 8);
      v24 = (v22 - 1) & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
      v25 = (_QWORD *)(v23 + 16LL * v24);
      v26 = (_BYTE *)*v25;
      if ( v9 == (_BYTE *)*v25 )
      {
LABEL_32:
        v27 = v25 + 1;
        goto LABEL_33;
      }
      v46 = 1;
      v30 = 0;
      while ( v26 != (_BYTE *)-4096LL )
      {
        if ( !v30 && v26 == (_BYTE *)-8192LL )
          v30 = v25;
        v24 = (v22 - 1) & (v46 + v24);
        v25 = (_QWORD *)(v23 + 16LL * v24);
        v26 = (_BYTE *)*v25;
        if ( v9 == (_BYTE *)*v25 )
          goto LABEL_32;
        ++v46;
      }
      v31 = *(_DWORD *)(a3 + 16);
      if ( !v30 )
        v30 = v25;
      ++*(_QWORD *)a3;
      v32 = v31 + 1;
      if ( 4 * (v31 + 1) < 3 * v22 )
      {
        if ( v22 - *(_DWORD *)(a3 + 20) - v32 > v22 >> 3 )
          goto LABEL_45;
        v48 = v21;
        sub_DA6D70(a3, v22);
        v40 = *(_DWORD *)(a3 + 24);
        if ( !v40 )
        {
LABEL_74:
          ++*(_DWORD *)(a3 + 16);
          BUG();
        }
        v41 = v40 - 1;
        v39 = 0;
        v42 = 1;
        v45 = *(_QWORD *)(a3 + 8);
        v43 = (v40 - 1) & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
        v30 = (_QWORD *)(v45 + 16 * v43);
        v44 = (_BYTE *)*v30;
        v32 = *(_DWORD *)(a3 + 16) + 1;
        v21 = v48;
        if ( v9 == (_BYTE *)*v30 )
          goto LABEL_45;
        while ( v44 != (_BYTE *)-4096LL )
        {
          if ( !v39 && v44 == (_BYTE *)-8192LL )
            v39 = v30;
          LODWORD(v43) = v41 & (v42 + v43);
          v30 = (_QWORD *)(v45 + 16LL * (unsigned int)v43);
          v44 = (_BYTE *)*v30;
          if ( v9 == (_BYTE *)*v30 )
            goto LABEL_45;
          ++v42;
        }
        goto LABEL_53;
      }
    }
    else
    {
      ++*(_QWORD *)a3;
    }
    v47 = v21;
    sub_DA6D70(a3, 2 * v22);
    v33 = *(_DWORD *)(a3 + 24);
    if ( !v33 )
      goto LABEL_74;
    v34 = v33 - 1;
    v35 = *(_QWORD *)(a3 + 8);
    v36 = (v33 - 1) & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
    v32 = *(_DWORD *)(a3 + 16) + 1;
    v21 = v47;
    v30 = (_QWORD *)(v35 + 16LL * v36);
    v37 = *v30;
    if ( v9 == (_BYTE *)*v30 )
      goto LABEL_45;
    v38 = 1;
    v39 = 0;
    while ( v37 != -4096 )
    {
      if ( !v39 && v37 == -8192 )
        v39 = v30;
      v36 = v34 & (v38 + v36);
      v30 = (_QWORD *)(v35 + 16LL * v36);
      v37 = *v30;
      if ( v9 == (_BYTE *)*v30 )
        goto LABEL_45;
      ++v38;
    }
LABEL_53:
    if ( v39 )
      v30 = v39;
LABEL_45:
    *(_DWORD *)(a3 + 16) = v32;
    if ( *v30 != -4096 )
      --*(_DWORD *)(a3 + 20);
    *v30 = v9;
    v27 = v30 + 1;
    *v27 = 0;
LABEL_33:
    *v27 = v21;
    if ( !v21 )
      return 0;
    goto LABEL_34;
  }
  return v8;
}
