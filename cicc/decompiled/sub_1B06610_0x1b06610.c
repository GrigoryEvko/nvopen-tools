// Function: sub_1B06610
// Address: 0x1b06610
//
_QWORD *__fastcall sub_1B06610(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v5; // rdx
  __int64 v6; // rbx
  _QWORD *result; // rax
  __int64 v8; // rcx
  unsigned int v9; // edi
  __int64 v10; // rdx
  __int64 v11; // rax
  __int64 v12; // rsi
  __int64 v13; // r12
  __int64 v14; // r15
  __int64 v15; // rax
  unsigned int v16; // esi
  int v17; // edx
  __int64 v18; // rax
  _QWORD *v19; // r14
  int v20; // edi
  __int64 v21; // rdx
  unsigned __int64 *v22; // rdi
  __int64 v23; // rdx
  __int64 v24; // r8
  unsigned int v25; // edx
  __int64 v26; // rdi
  __int64 v27; // rax
  __int64 v28; // rdx
  _QWORD *v29; // r15
  __int64 v30; // rsi
  unsigned __int64 v31; // rdx
  __int64 v32; // rsi
  __int64 v33; // rdx
  int v34; // r11d
  __int64 v35; // r10
  int v36; // esi
  _QWORD *v37; // rdx
  unsigned int v38; // r9d
  __int64 v39; // rdi
  int v40; // r10d
  _QWORD *v41; // r11
  int v42; // edi
  int v43; // edx
  int v44; // r11d
  __int64 v45; // r10
  int v46; // esi
  unsigned int v47; // r9d
  __int64 v48; // rdi
  __int64 v49; // [rsp+8h] [rbp-88h]
  __int64 v50; // [rsp+10h] [rbp-80h]
  __int64 v51; // [rsp+10h] [rbp-80h]
  __int64 v52; // [rsp+10h] [rbp-80h]
  __int64 v53; // [rsp+10h] [rbp-80h]
  __int64 v54; // [rsp+10h] [rbp-80h]
  __int64 v57; // [rsp+28h] [rbp-68h]
  _QWORD v58[2]; // [rsp+38h] [rbp-58h] BYREF
  __int64 v59; // [rsp+48h] [rbp-48h]
  __int64 v60; // [rsp+50h] [rbp-40h]

  v6 = sub_157F280(a1);
  result = v58;
  if ( v6 != v5 )
  {
    v8 = a4;
    v57 = v5;
    while ( 1 )
    {
      v9 = *(_DWORD *)(v6 + 20) & 0xFFFFFFF;
      if ( v9 )
        break;
LABEL_43:
      result = *(_QWORD **)(v6 + 32);
      if ( !result )
        BUG();
      v6 = 0;
      if ( *((_BYTE *)result - 8) == 77 )
        v6 = (__int64)(result - 3);
      if ( v57 == v6 )
        return result;
    }
    v10 = 24LL * *(unsigned int *)(v6 + 56) + 8;
    v11 = 0;
    while ( 1 )
    {
      v12 = v6 - 24LL * v9;
      if ( (*(_BYTE *)(v6 + 23) & 0x40) != 0 )
        v12 = *(_QWORD *)(v6 - 8);
      v13 = 8 * v11;
      if ( *(_QWORD *)(v12 + v10) == a2 )
        break;
      ++v11;
      v10 += 8;
      if ( v9 <= (unsigned int)v11 )
        goto LABEL_43;
    }
    v14 = 24 * v11;
    v15 = *(_QWORD *)(v12 + 24 * v11);
    v58[0] = 2;
    v58[1] = 0;
    if ( v15 )
    {
      v59 = v15;
      if ( v15 != -16 && v15 != -8 )
      {
        v50 = v8;
        sub_164C220((__int64)v58);
        v8 = v50;
      }
    }
    else
    {
      v59 = 0;
    }
    v16 = *(_DWORD *)(v8 + 24);
    v60 = v8;
    if ( v16 )
    {
      v18 = v59;
      v24 = *(_QWORD *)(v8 + 8);
      v25 = (v16 - 1) & (((unsigned int)v59 >> 9) ^ ((unsigned int)v59 >> 4));
      v19 = (_QWORD *)(v24 + ((unsigned __int64)v25 << 6));
      v26 = v19[3];
      if ( v59 == v26 )
      {
LABEL_28:
        if ( v18 != -8 && v18 != 0 && v18 != -16 )
        {
          v53 = v8;
          sub_1649B30(v58);
          v8 = v53;
        }
        v27 = v19[7];
        if ( v27 )
        {
          if ( (*(_BYTE *)(v6 + 23) & 0x40) != 0 )
            v28 = *(_QWORD *)(v6 - 8);
          else
            v28 = v6 - 24LL * (*(_DWORD *)(v6 + 20) & 0xFFFFFFF);
          v29 = (_QWORD *)(v28 + v14);
          if ( *v29 )
          {
            v30 = v29[1];
            v31 = v29[2] & 0xFFFFFFFFFFFFFFFCLL;
            *(_QWORD *)v31 = v30;
            if ( v30 )
              *(_QWORD *)(v30 + 16) = *(_QWORD *)(v30 + 16) & 3LL | v31;
          }
          *v29 = v27;
          v32 = *(_QWORD *)(v27 + 8);
          v29[1] = v32;
          if ( v32 )
            *(_QWORD *)(v32 + 16) = (unsigned __int64)(v29 + 1) | *(_QWORD *)(v32 + 16) & 3LL;
          v29[2] = v29[2] & 3LL | (v27 + 8);
          *(_QWORD *)(v27 + 8) = v29;
        }
        if ( (*(_BYTE *)(v6 + 23) & 0x40) != 0 )
          v33 = *(_QWORD *)(v6 - 8);
        else
          v33 = v6 - 24LL * (*(_DWORD *)(v6 + 20) & 0xFFFFFFF);
        *(_QWORD *)(v13 + v33 + 24LL * *(unsigned int *)(v6 + 56) + 8) = a3;
        goto LABEL_43;
      }
      v40 = 1;
      v41 = 0;
      while ( v26 != -8 )
      {
        if ( !v41 && v26 == -16 )
          v41 = v19;
        v25 = (v16 - 1) & (v40 + v25);
        v19 = (_QWORD *)(v24 + ((unsigned __int64)v25 << 6));
        v26 = v19[3];
        if ( v59 == v26 )
          goto LABEL_28;
        ++v40;
      }
      v42 = *(_DWORD *)(v8 + 16);
      if ( v41 )
        v19 = v41;
      ++*(_QWORD *)v8;
      v20 = v42 + 1;
      if ( 4 * v20 < 3 * v16 )
      {
        if ( v16 - *(_DWORD *)(v8 + 20) - v20 > v16 >> 3 )
        {
LABEL_18:
          *(_DWORD *)(v8 + 16) = v20;
          if ( v19[3] == -8 )
          {
            v22 = v19 + 1;
            if ( v18 != -8 )
            {
LABEL_23:
              v19[3] = v18;
              if ( v18 == 0 || v18 == -8 || v18 == -16 )
              {
                v18 = v59;
              }
              else
              {
                v52 = v8;
                sub_1649AC0(v22, v58[0] & 0xFFFFFFFFFFFFFFF8LL);
                v18 = v59;
                v8 = v52;
              }
            }
          }
          else
          {
            --*(_DWORD *)(v8 + 20);
            v21 = v19[3];
            if ( v18 != v21 )
            {
              v22 = v19 + 1;
              if ( v21 != -8 && v21 != 0 && v21 != -16 )
              {
                v49 = v8;
                sub_1649B30(v22);
                v18 = v59;
                v8 = v49;
              }
              goto LABEL_23;
            }
          }
          v23 = v60;
          v19[5] = 6;
          v19[6] = 0;
          v19[4] = v23;
          v19[7] = 0;
          goto LABEL_28;
        }
        v54 = v8;
        sub_12E48B0(v8, v16);
        v8 = v54;
        v43 = *(_DWORD *)(v54 + 24);
        if ( !v43 )
          goto LABEL_16;
        v18 = v59;
        v44 = v43 - 1;
        v45 = *(_QWORD *)(v54 + 8);
        v46 = 1;
        v37 = 0;
        v47 = v44 & (((unsigned int)v59 >> 9) ^ ((unsigned int)v59 >> 4));
        v19 = (_QWORD *)(v45 + ((unsigned __int64)v47 << 6));
        v48 = v19[3];
        if ( v48 == v59 )
          goto LABEL_17;
        while ( v48 != -8 )
        {
          if ( !v37 && v48 == -16 )
            v37 = v19;
          v47 = v44 & (v46 + v47);
          v19 = (_QWORD *)(v45 + ((unsigned __int64)v47 << 6));
          v48 = v19[3];
          if ( v59 == v48 )
            goto LABEL_17;
          ++v46;
        }
        goto LABEL_70;
      }
    }
    else
    {
      ++*(_QWORD *)v8;
    }
    v51 = v8;
    sub_12E48B0(v8, 2 * v16);
    v8 = v51;
    v17 = *(_DWORD *)(v51 + 24);
    if ( !v17 )
    {
LABEL_16:
      v18 = v59;
      v19 = 0;
LABEL_17:
      v20 = *(_DWORD *)(v8 + 16) + 1;
      goto LABEL_18;
    }
    v18 = v59;
    v34 = v17 - 1;
    v35 = *(_QWORD *)(v51 + 8);
    v36 = 1;
    v37 = 0;
    v38 = v34 & (((unsigned int)v59 >> 9) ^ ((unsigned int)v59 >> 4));
    v19 = (_QWORD *)(v35 + ((unsigned __int64)v38 << 6));
    v39 = v19[3];
    if ( v39 == v59 )
      goto LABEL_17;
    while ( v39 != -8 )
    {
      if ( v39 == -16 && !v37 )
        v37 = v19;
      v38 = v34 & (v36 + v38);
      v19 = (_QWORD *)(v35 + ((unsigned __int64)v38 << 6));
      v39 = v19[3];
      if ( v59 == v39 )
        goto LABEL_17;
      ++v36;
    }
LABEL_70:
    if ( v37 )
      v19 = v37;
    goto LABEL_17;
  }
  return result;
}
