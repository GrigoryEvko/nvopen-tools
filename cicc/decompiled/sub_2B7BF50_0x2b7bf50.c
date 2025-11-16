// Function: sub_2B7BF50
// Address: 0x2b7bf50
//
__int64 __fastcall sub_2B7BF50(__int64 a1, __int64 a2, char *a3, unsigned __int64 a4)
{
  __int64 v6; // rax
  __int64 v7; // r8
  __int64 v8; // r9
  __int64 v9; // r11
  __int64 v10; // r13
  __int64 *v11; // r14
  __int64 result; // rax
  __int64 *v13; // r8
  __int64 v14; // r9
  int v15; // r11d
  __int64 *v16; // r15
  unsigned int v17; // esi
  __int64 *v18; // rcx
  unsigned int v19; // edi
  __int64 v20; // rax
  unsigned int v21; // r11d
  int v22; // ecx
  _DWORD *v23; // rdx
  __int64 v24; // r10
  __int64 v25; // rdi
  __int64 v26; // rcx
  __int64 v27; // rax
  int *v28; // rdx
  __int64 v29; // r9
  __int64 v30; // rsi
  __int64 ***v31; // rax
  __int64 v32; // rcx
  __int64 v33; // rsi
  __int64 v34; // rax
  _DWORD *v35; // rdx
  unsigned __int64 v36; // r9
  __int64 v37; // rdx
  _DWORD *v38; // rcx
  int v39; // r8d
  int v40; // r8d
  __int64 v41; // rax
  __int64 v42; // r9
  __int64 v43; // rdx
  __int64 ***v44; // rax
  __int64 v45; // rdi
  signed __int64 v46; // r14
  unsigned __int64 v47; // rdx
  unsigned __int64 v48; // r13
  __int64 v49; // rdx
  __int64 v50; // rax
  __int64 v51; // rcx
  __int64 v52; // r11
  int *v53; // rdx
  __int64 v54; // rsi
  __int64 v55; // r9
  __int64 ***v56; // rax
  __int64 v57; // rcx
  __int64 v58; // rsi
  __int64 v59; // rax
  _DWORD *v60; // rdx
  __int64 v61; // [rsp+8h] [rbp-68h]
  __int64 v63; // [rsp+18h] [rbp-58h] BYREF
  __int64 v64; // [rsp+20h] [rbp-50h] BYREF
  __int64 v65; // [rsp+28h] [rbp-48h]
  __int64 v66; // [rsp+30h] [rbp-40h]
  __int64 v67; // [rsp+38h] [rbp-38h]

  v63 = a2;
  v6 = sub_2B2EA50((_QWORD *)a1, a2, 0);
  v9 = *(unsigned int *)(a1 + 88);
  v63 = v6;
  v10 = v6;
  if ( !(_DWORD)v9 )
  {
    if ( !*(_DWORD *)(a1 + 92) )
    {
      sub_C8D5F0(a1 + 80, (const void *)(a1 + 96), 1u, 8u, v7, v8);
      v9 = *(unsigned int *)(a1 + 88);
    }
    v45 = 0;
    v46 = 4 * a4;
    *(_QWORD *)(*(_QWORD *)(a1 + 80) + 8 * v9) = v10;
    v47 = *(unsigned int *)(a1 + 28);
    result = 0;
    ++*(_DWORD *)(a1 + 88);
    v48 = (__int64)(4 * a4) >> 2;
    *(_DWORD *)(a1 + 24) = 0;
    if ( v48 > v47 )
    {
      sub_C8D5F0(a1 + 16, (const void *)(a1 + 32), v46 >> 2, 4u, v7, v8);
      result = *(unsigned int *)(a1 + 24);
      v45 = 4 * result;
    }
    if ( v46 )
    {
      memcpy((void *)(*(_QWORD *)(a1 + 16) + v45), a3, v46);
      result = *(unsigned int *)(a1 + 24);
    }
    *(_DWORD *)(a1 + 24) = result + v48;
    return result;
  }
  v11 = *(__int64 **)(a1 + 80);
  result = (__int64)sub_2B0BE70(v11, (__int64)&v11[(unsigned int)v9], &v63);
  v16 = (__int64 *)result;
  if ( v13 != (__int64 *)result )
  {
LABEL_3:
    v17 = *(_DWORD *)(a1 + 24);
    if ( v11 == v13 )
    {
      v19 = 0;
      goto LABEL_12;
    }
    result = *(_QWORD *)(*v11 + 8);
LABEL_5:
    v18 = v11 + 1;
    v19 = 0;
    while ( 1 )
    {
      v21 = 1;
      if ( *(_BYTE *)(*(_QWORD *)a1 + 8LL) == 17 )
        v21 = *(_DWORD *)(*(_QWORD *)a1 + 32LL);
      result = *(_DWORD *)(result + 32) / v21;
      if ( v19 < (unsigned int)result )
        v19 = result;
      if ( v13 == v18 )
        break;
      v20 = *v18++;
      result = *(_QWORD *)(v20 + 8);
    }
LABEL_12:
    if ( !v17 )
      return result;
    goto LABEL_13;
  }
  v17 = *(_DWORD *)(a1 + 24);
  v24 = *v11;
  if ( v15 == 2 )
  {
    v50 = *(_QWORD *)(a1 + 120);
    v51 = v17;
    v52 = *(__int64 *)((char *)v11 + v14 - 8);
    v53 = *(int **)(a1 + 16);
    v54 = *(_QWORD *)(v50 + 3344);
    v55 = *(_QWORD *)a1;
    v64 = *(_QWORD *)(a1 + 112);
    v65 = v50 + 3112;
    v67 = v54;
    v66 = v50 + 3160;
    v56 = sub_2B7A630(v24, v52, v53, v51, (__int64)&v64, v55);
    v57 = *(unsigned int *)(a1 + 24);
    v58 = *(_QWORD *)(a1 + 16);
    v24 = (__int64)v56;
    v59 = 0;
    if ( (_DWORD)v57 )
    {
      do
      {
        v60 = (_DWORD *)(v58 + 4LL * (unsigned int)v59);
        if ( *v60 != -1 )
          *v60 = v59;
        ++v59;
      }
      while ( v57 != v59 );
LABEL_28:
      v36 = *(unsigned int *)(a1 + 24);
      v25 = *(_QWORD *)(v10 + 8);
      v17 = *(_DWORD *)(a1 + 24);
      if ( a4 >= v36 )
        LODWORD(v36) = a4;
      result = *(_QWORD *)(v24 + 8);
      if ( !v17 )
        goto LABEL_38;
      goto LABEL_31;
    }
LABEL_69:
    result = *(_QWORD *)(v24 + 8);
    v25 = *(_QWORD *)(v10 + 8);
    goto LABEL_38;
  }
  result = *(_QWORD *)(v24 + 8);
  v25 = *(_QWORD *)(v10 + 8);
  if ( result != v25 )
  {
    v26 = v17;
    if ( *(_DWORD *)(result + 32) == v17 )
    {
      if ( v17 < a4 )
        LODWORD(v26) = a4;
      LODWORD(v36) = v26;
      if ( !v17 )
      {
LABEL_39:
        v41 = *(_QWORD *)(a1 + 120);
        v42 = *(_QWORD *)a1;
        v61 = v24;
        v43 = *(_QWORD *)(v41 + 3344);
        v64 = *(_QWORD *)(a1 + 112);
        v65 = v41 + 3112;
        v67 = v43;
        v66 = v41 + 3160;
        v44 = sub_2B7A630(v10, 0, (int *)a3, a4, (__int64)&v64, v42);
        v24 = v61;
        v63 = (__int64)v44;
        v10 = (__int64)v44;
LABEL_40:
        **(_QWORD **)(a1 + 80) = v24;
        if ( *(_DWORD *)(a1 + 88) != 2 )
          return sub_94F890(a1 + 80, v10);
        result = *(_QWORD *)(a1 + 80);
        *(_QWORD *)(result + 8) = v10;
        return result;
      }
LABEL_31:
      v37 = 0;
      do
      {
        while ( 1 )
        {
          v38 = (_DWORD *)(*(_QWORD *)(a1 + 16) + 4 * v37);
          if ( *v38 == -1 )
          {
            v39 = *(_DWORD *)&a3[4 * v37];
            if ( v39 != -1 )
              break;
          }
          if ( v17 <= (unsigned int)++v37 )
            goto LABEL_38;
        }
        if ( result == v25 )
          v40 = *(_DWORD *)(result + 32) + v39;
        else
          v40 = v36 + v37;
        ++v37;
        *v38 = v40;
        result = *(_QWORD *)(v24 + 8);
        v25 = *(_QWORD *)(v10 + 8);
      }
      while ( v17 > (unsigned int)v37 );
LABEL_38:
      if ( result == v25 )
        goto LABEL_40;
      goto LABEL_39;
    }
    v27 = *(_QWORD *)(a1 + 120);
    v28 = *(int **)(a1 + 16);
    v29 = *(_QWORD *)a1;
    v30 = *(_QWORD *)(v27 + 3344);
    v64 = *(_QWORD *)(a1 + 112);
    v65 = v27 + 3112;
    v67 = v30;
    v66 = v27 + 3160;
    v31 = sub_2B7A630(v24, 0, v28, v26, (__int64)&v64, v29);
    v32 = *(unsigned int *)(a1 + 24);
    v33 = *(_QWORD *)(a1 + 16);
    v24 = (__int64)v31;
    v34 = 0;
    if ( (_DWORD)v32 )
    {
      do
      {
        v35 = (_DWORD *)(v33 + 4LL * (unsigned int)v34);
        if ( *v35 != -1 )
          *v35 = v34;
        ++v34;
      }
      while ( v32 != v34 );
      goto LABEL_28;
    }
    goto LABEL_69;
  }
  if ( !v17 )
  {
    if ( v11 == v13 )
      return result;
    goto LABEL_5;
  }
  v49 = 0;
  do
  {
    if ( *(_DWORD *)&a3[v49] != -1 && *(_DWORD *)(*(_QWORD *)(a1 + 16) + v49) == -1 )
    {
      sub_94F890(a1 + 80, v10);
      v11 = *(__int64 **)(a1 + 80);
      result = *(unsigned int *)(a1 + 88);
      v13 = &v11[result];
      goto LABEL_3;
    }
    v49 += 4;
  }
  while ( 4LL * v17 != v49 );
  if ( v11 != v16 )
  {
    v13 = v16;
    goto LABEL_5;
  }
  v19 = 0;
LABEL_13:
  result = 0;
  do
  {
    while ( 1 )
    {
      v22 = *(_DWORD *)&a3[4 * result];
      if ( v22 != -1 )
      {
        v23 = (_DWORD *)(*(_QWORD *)(a1 + 16) + 4 * result);
        if ( *v23 == -1 )
          break;
      }
      if ( v17 <= (unsigned int)++result )
        return result;
    }
    if ( *(__int64 **)(a1 + 80) != v16 )
      v22 += v19;
    ++result;
    *v23 = v22;
  }
  while ( v17 > (unsigned int)result );
  return result;
}
