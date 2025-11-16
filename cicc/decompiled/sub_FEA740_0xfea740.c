// Function: sub_FEA740
// Address: 0xfea740
//
unsigned int *__fastcall sub_FEA740(__int64 a1, unsigned int *a2, __int64 a3, __int64 a4)
{
  __int64 v6; // r12
  __int64 *v7; // r14
  __int64 v8; // rax
  _DWORD *v9; // rdi
  unsigned __int64 *v10; // r12
  unsigned __int64 v11; // r12
  unsigned int *result; // rax
  __int64 v13; // rdx
  __int64 v14; // r9
  unsigned int *v15; // r15
  unsigned int v16; // r14d
  __int64 v17; // rcx
  unsigned __int64 v18; // rax
  unsigned __int64 *v19; // rcx
  __int64 v20; // r13
  unsigned int v21; // edx
  unsigned __int64 v22; // rax
  __int64 v23; // r8
  bool v24; // cf
  unsigned __int64 v25; // rbx
  int v26; // eax
  __int64 v27; // rdx
  __int64 *v28; // rcx
  __int64 v29; // rax
  _DWORD *v30; // rdi
  __int64 v31; // r11
  __int64 v32; // rax
  __int64 *v33; // rdx
  __int64 v34; // rax
  __int64 v35; // rdx
  _DWORD *v36; // rax
  __int64 v37; // r11
  __int64 v38; // rbx
  bool v39; // al
  __int64 v40; // r13
  __int64 v41; // rax
  bool v42; // al
  __int64 *v43; // [rsp+0h] [rbp-80h]
  __int64 v44; // [rsp+0h] [rbp-80h]
  __int64 v45; // [rsp+8h] [rbp-78h]
  unsigned __int64 v46; // [rsp+8h] [rbp-78h]
  __int64 *v47; // [rsp+8h] [rbp-78h]
  unsigned int *v49; // [rsp+28h] [rbp-58h]
  unsigned __int64 v50; // [rsp+30h] [rbp-50h]
  __int64 v51; // [rsp+38h] [rbp-48h]
  __int64 v52; // [rsp+38h] [rbp-48h]
  __int64 v53; // [rsp+38h] [rbp-48h]
  unsigned int v54; // [rsp+48h] [rbp-38h] BYREF
  unsigned int v55[13]; // [rsp+4Ch] [rbp-34h] BYREF

  v6 = *(_QWORD *)(a1 + 64) + 24LL * *a2;
  v7 = *(__int64 **)(v6 + 8);
  if ( !v7 )
    goto LABEL_4;
  v8 = *((unsigned int *)v7 + 3);
  v9 = (_DWORD *)v7[12];
  if ( (unsigned int)v8 > 1 )
  {
    if ( !sub_FDC990(v9, &v9[v8], (_DWORD *)v6) )
      goto LABEL_4;
  }
  else if ( *(_DWORD *)v6 != *v9 )
  {
LABEL_4:
    v10 = (unsigned __int64 *)(v6 + 16);
    goto LABEL_5;
  }
  if ( !*((_BYTE *)v7 + 8) )
    goto LABEL_4;
  v40 = *v7;
  if ( !*v7
    || (v41 = *(unsigned int *)(v40 + 12), (unsigned int)v41 <= 1)
    || !sub_FDC990(*(_DWORD **)(v40 + 96), (_DWORD *)(*(_QWORD *)(v40 + 96) + 4 * v41), (_DWORD *)v6)
    || (v10 = (unsigned __int64 *)(v40 + 152), !*(_BYTE *)(v40 + 8)) )
  {
    v10 = (unsigned __int64 *)(v7 + 19);
  }
LABEL_5:
  v11 = *v10;
  sub_FE9FC0(a4);
  result = *(unsigned int **)a4;
  v13 = 4LL * *(unsigned int *)(a4 + 8);
  v49 = (unsigned int *)(*(_QWORD *)a4 + v13 * 4);
  if ( &result[v13] != result )
  {
    v14 = a3;
    v15 = result + 1;
    v16 = *(_DWORD *)(a4 + 80);
    while ( 1 )
    {
      v20 = *(_QWORD *)(v15 + 1);
      v21 = v16;
      v51 = v14;
      v16 -= v20;
      sub_F02DB0(v55, v20, v21);
      v54 = v55[0];
      v22 = sub_F02E20(&v54, v11);
      v14 = v51;
      v24 = v11 < v22;
      v11 -= v22;
      v25 = v22;
      if ( v24 )
        v11 = 0;
      v26 = *(v15 - 1);
      if ( !v26 )
        break;
      if ( v26 == 2 )
      {
        v34 = *(unsigned int *)(v51 + 12);
        v35 = 0;
        if ( (unsigned int)v34 > 1 )
        {
          v36 = sub_FE8370(*(_DWORD **)(v51 + 96), *(_QWORD *)(v51 + 96) + 4 * v34, v15);
          v35 = 8 * (((__int64)v36 - v37) >> 2);
        }
        v33 = (__int64 *)(*(_QWORD *)(v14 + 128) + v35);
LABEL_25:
        v24 = __CFADD__(*v33, v25);
        v38 = *v33 + v25;
        if ( v24 )
          v38 = -1;
        result = v15 + 4;
        *v33 = v38;
        if ( v49 == v15 + 3 )
          return result;
        goto LABEL_11;
      }
      v17 = *(unsigned int *)(v51 + 24);
      v18 = v50 & 0xFFFFFFFF00000000LL | *v15;
      v50 = v18;
      if ( v17 + 1 > (unsigned __int64)*(unsigned int *)(v51 + 28) )
      {
        v46 = v18;
        sub_C8D5F0(v51 + 16, (const void *)(v51 + 32), v17 + 1, 0x10u, v23, v51);
        v14 = v51;
        v18 = v46;
        v17 = *(unsigned int *)(v51 + 24);
      }
      v19 = (unsigned __int64 *)(*(_QWORD *)(v14 + 16) + 16 * v17);
      *v19 = v18;
      result = v15 + 4;
      v19[1] = v25;
      ++*(_DWORD *)(v14 + 24);
      if ( v49 == v15 + 3 )
        return result;
LABEL_11:
      v15 = result;
    }
    v27 = *(_QWORD *)(a1 + 64) + 24LL * *v15;
    v28 = *(__int64 **)(v27 + 8);
    if ( v28 )
    {
      v29 = *((unsigned int *)v28 + 3);
      v30 = (_DWORD *)v28[12];
      if ( (unsigned int)v29 > 1 )
      {
        v45 = v51;
        v52 = *(_QWORD *)(a1 + 64) + 24LL * *v15;
        v43 = *(__int64 **)(v27 + 8);
        v39 = sub_FDC990(v30, &v30[v29], (_DWORD *)v27);
        v27 = v52;
        v14 = v45;
        if ( v39 )
        {
          v28 = v43;
          if ( *((_BYTE *)v43 + 8) )
          {
LABEL_19:
            v31 = *v28;
            if ( *v28
              && (v32 = *(unsigned int *)(v31 + 12), (unsigned int)v32 > 1)
              && (v44 = v14,
                  v47 = v28,
                  v53 = *v28,
                  v42 = sub_FDC990(*(_DWORD **)(v31 + 96), (_DWORD *)(*(_QWORD *)(v31 + 96) + 4 * v32), (_DWORD *)v27),
                  v28 = v47,
                  v14 = v44,
                  v42) )
            {
              v33 = (__int64 *)(v53 + 152);
              if ( !*(_BYTE *)(v53 + 8) )
                v33 = v47 + 19;
            }
            else
            {
              v33 = v28 + 19;
            }
            goto LABEL_25;
          }
        }
      }
      else if ( *(_DWORD *)v27 == *v30 && *((_BYTE *)v28 + 8) )
      {
        goto LABEL_19;
      }
    }
    v33 = (__int64 *)(v27 + 16);
    goto LABEL_25;
  }
  return result;
}
