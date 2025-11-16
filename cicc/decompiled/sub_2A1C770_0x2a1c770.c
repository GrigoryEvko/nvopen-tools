// Function: sub_2A1C770
// Address: 0x2a1c770
//
_QWORD *__fastcall sub_2A1C770(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rdx
  __int64 v7; // r14
  _QWORD *result; // rax
  __int64 v9; // rcx
  __int64 v10; // rbx
  __int64 v11; // r12
  __int64 v12; // rbx
  __int64 v13; // rax
  unsigned int v14; // esi
  int v15; // edx
  __int64 v16; // rax
  _QWORD *v17; // rcx
  int v18; // edi
  __int64 v19; // rdx
  unsigned __int64 *v20; // rdi
  __int64 v21; // rdx
  _QWORD *v22; // rdx
  __int64 v23; // r8
  __int64 v24; // r10
  _QWORD *v25; // rdx
  __int64 v26; // rdi
  __int64 v27; // rax
  __int64 v28; // rbx
  __int64 v29; // rdx
  __int64 v30; // rdx
  int v31; // r11d
  __int64 v32; // r10
  int v33; // esi
  _QWORD *v34; // rdx
  unsigned int v35; // r9d
  __int64 v36; // rdi
  int v37; // r11d
  int v38; // edi
  int v39; // edx
  int v40; // r11d
  __int64 v41; // r10
  int v42; // esi
  unsigned int v43; // r9d
  __int64 v44; // rdi
  _QWORD *v45; // [rsp+8h] [rbp-88h]
  _QWORD *v46; // [rsp+10h] [rbp-80h]
  _QWORD *v47; // [rsp+10h] [rbp-80h]
  __int64 v49; // [rsp+28h] [rbp-68h]
  _QWORD v50[2]; // [rsp+38h] [rbp-58h] BYREF
  __int64 v51; // [rsp+48h] [rbp-48h]
  __int64 v52; // [rsp+50h] [rbp-40h]

  v7 = sub_AA5930(a1);
  result = v50;
  if ( v7 != v6 )
  {
    v49 = v6;
    while ( (*(_DWORD *)(v7 + 4) & 0x7FFFFFF) == 0 )
    {
LABEL_38:
      result = *(_QWORD **)(v7 + 32);
      if ( !result )
        BUG();
      v7 = 0;
      if ( *((_BYTE *)result - 24) == 84 )
        v7 = (__int64)(result - 3);
      if ( v49 == v7 )
        return result;
    }
    v9 = *(_QWORD *)(v7 - 8);
    v10 = 0;
    while ( 1 )
    {
      v11 = 8 * v10;
      if ( *(_QWORD *)(v9 + 32LL * *(unsigned int *)(v7 + 72) + 8 * v10) == a2 )
        break;
      if ( (*(_DWORD *)(v7 + 4) & 0x7FFFFFFu) <= (unsigned int)++v10 )
        goto LABEL_38;
    }
    v12 = 32 * v10;
    v13 = *(_QWORD *)(v9 + v12);
    v50[0] = 2;
    v50[1] = 0;
    if ( v13 )
    {
      v51 = v13;
      if ( v13 != -8192 && v13 != -4096 )
        sub_BD73F0((__int64)v50);
    }
    else
    {
      v51 = 0;
    }
    v14 = *(_DWORD *)(a4 + 24);
    v52 = a4;
    if ( v14 )
    {
      v16 = v51;
      v23 = *(_QWORD *)(a4 + 8);
      v24 = (v14 - 1) & (((unsigned int)v51 >> 9) ^ ((unsigned int)v51 >> 4));
      v25 = (_QWORD *)(v23 + (v24 << 6));
      v26 = v25[3];
      if ( v51 == v26 )
      {
LABEL_26:
        v22 = v25 + 5;
LABEL_27:
        if ( v16 != -4096 && v16 != 0 && v16 != -8192 )
        {
          v47 = v22;
          sub_BD60C0(v50);
          v22 = v47;
        }
        v27 = v22[2];
        if ( v27 )
        {
          v28 = *(_QWORD *)(v7 - 8) + v12;
          if ( *(_QWORD *)v28 )
          {
            v29 = *(_QWORD *)(v28 + 8);
            **(_QWORD **)(v28 + 16) = v29;
            if ( v29 )
              *(_QWORD *)(v29 + 16) = *(_QWORD *)(v28 + 16);
          }
          *(_QWORD *)v28 = v27;
          v30 = *(_QWORD *)(v27 + 16);
          *(_QWORD *)(v28 + 8) = v30;
          if ( v30 )
            *(_QWORD *)(v30 + 16) = v28 + 8;
          *(_QWORD *)(v28 + 16) = v27 + 16;
          *(_QWORD *)(v27 + 16) = v28;
        }
        *(_QWORD *)(*(_QWORD *)(v7 - 8) + 32LL * *(unsigned int *)(v7 + 72) + v11) = a3;
        goto LABEL_38;
      }
      v37 = 1;
      v17 = 0;
      while ( v26 != -4096 )
      {
        if ( !v17 && v26 == -8192 )
          v17 = v25;
        LODWORD(v24) = (v14 - 1) & (v37 + v24);
        v25 = (_QWORD *)(v23 + ((unsigned __int64)(unsigned int)v24 << 6));
        v26 = v25[3];
        if ( v51 == v26 )
          goto LABEL_26;
        ++v37;
      }
      v38 = *(_DWORD *)(a4 + 16);
      if ( !v17 )
        v17 = v25;
      ++*(_QWORD *)a4;
      v18 = v38 + 1;
      if ( 4 * v18 < 3 * v14 )
      {
        if ( v14 - *(_DWORD *)(a4 + 20) - v18 > v14 >> 3 )
        {
LABEL_16:
          *(_DWORD *)(a4 + 16) = v18;
          if ( v17[3] == -4096 )
          {
            v20 = v17 + 1;
            if ( v16 != -4096 )
            {
LABEL_21:
              v17[3] = v16;
              if ( v16 == 0 || v16 == -4096 || v16 == -8192 )
              {
                v16 = v51;
              }
              else
              {
                v46 = v17;
                sub_BD6050(v20, v50[0] & 0xFFFFFFFFFFFFFFF8LL);
                v16 = v51;
                v17 = v46;
              }
            }
          }
          else
          {
            --*(_DWORD *)(a4 + 20);
            v19 = v17[3];
            if ( v16 != v19 )
            {
              v20 = v17 + 1;
              if ( v19 != -4096 && v19 != 0 && v19 != -8192 )
              {
                v45 = v17;
                sub_BD60C0(v20);
                v16 = v51;
                v17 = v45;
              }
              goto LABEL_21;
            }
          }
          v21 = v52;
          v17[5] = 6;
          v17[6] = 0;
          v17[4] = v21;
          v22 = v17 + 5;
          v17[7] = 0;
          goto LABEL_27;
        }
        sub_CF32C0(a4, v14);
        v39 = *(_DWORD *)(a4 + 24);
        if ( !v39 )
          goto LABEL_14;
        v16 = v51;
        v40 = v39 - 1;
        v41 = *(_QWORD *)(a4 + 8);
        v42 = 1;
        v34 = 0;
        v43 = v40 & (((unsigned int)v51 >> 9) ^ ((unsigned int)v51 >> 4));
        v17 = (_QWORD *)(v41 + ((unsigned __int64)v43 << 6));
        v44 = v17[3];
        if ( v44 == v51 )
          goto LABEL_15;
        while ( v44 != -4096 )
        {
          if ( v44 == -8192 && !v34 )
            v34 = v17;
          v43 = v40 & (v42 + v43);
          v17 = (_QWORD *)(v41 + ((unsigned __int64)v43 << 6));
          v44 = v17[3];
          if ( v51 == v44 )
            goto LABEL_15;
          ++v42;
        }
        goto LABEL_63;
      }
    }
    else
    {
      ++*(_QWORD *)a4;
    }
    sub_CF32C0(a4, 2 * v14);
    v15 = *(_DWORD *)(a4 + 24);
    if ( !v15 )
    {
LABEL_14:
      v16 = v51;
      v17 = 0;
LABEL_15:
      v18 = *(_DWORD *)(a4 + 16) + 1;
      goto LABEL_16;
    }
    v16 = v51;
    v31 = v15 - 1;
    v32 = *(_QWORD *)(a4 + 8);
    v33 = 1;
    v34 = 0;
    v35 = v31 & (((unsigned int)v51 >> 9) ^ ((unsigned int)v51 >> 4));
    v17 = (_QWORD *)(v32 + ((unsigned __int64)v35 << 6));
    v36 = v17[3];
    if ( v51 == v36 )
      goto LABEL_15;
    while ( v36 != -4096 )
    {
      if ( !v34 && v36 == -8192 )
        v34 = v17;
      v35 = v31 & (v33 + v35);
      v17 = (_QWORD *)(v32 + ((unsigned __int64)v35 << 6));
      v36 = v17[3];
      if ( v51 == v36 )
        goto LABEL_15;
      ++v33;
    }
LABEL_63:
    if ( v34 )
      v17 = v34;
    goto LABEL_15;
  }
  return result;
}
