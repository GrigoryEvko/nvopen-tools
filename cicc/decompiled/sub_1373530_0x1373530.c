// Function: sub_1373530
// Address: 0x1373530
//
unsigned int *__fastcall sub_1373530(__int64 a1, unsigned int *a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rax
  __int64 v7; // r13
  __int64 *v8; // rbx
  __int64 v9; // rax
  _DWORD *v10; // rdi
  unsigned __int64 *v11; // r13
  unsigned __int64 v12; // rbx
  unsigned int *result; // rax
  unsigned int v14; // r13d
  __int64 v15; // r12
  unsigned int *v16; // rsi
  unsigned int *v17; // r12
  unsigned __int64 v18; // xmm0_8
  __int64 v19; // rax
  unsigned __int64 *v20; // rax
  __int64 v21; // r14
  __int64 v22; // rdx
  unsigned __int64 v23; // rax
  bool v24; // cf
  int v25; // eax
  __int64 v26; // rdx
  __int64 v27; // rcx
  __int64 v28; // rax
  _DWORD *v29; // rdi
  __int64 v30; // r9
  __int64 v31; // rax
  __int64 *v32; // rdx
  __int64 v33; // rax
  __int64 v34; // rdx
  _DWORD *v35; // r9
  _DWORD *v36; // rax
  __int64 v37; // r9
  __int64 v38; // r8
  bool v39; // al
  unsigned __int64 v40; // r8
  __int64 v41; // r14
  __int64 v42; // rax
  _DWORD *v43; // rdi
  bool v44; // al
  __int64 v45; // [rsp+8h] [rbp-78h]
  __int64 v46; // [rsp+8h] [rbp-78h]
  unsigned __int64 v47; // [rsp+10h] [rbp-70h]
  __int64 v48; // [rsp+18h] [rbp-68h]
  __int64 v49; // [rsp+20h] [rbp-60h]
  int *v50; // [rsp+28h] [rbp-58h]
  int *v51; // [rsp+30h] [rbp-50h]
  unsigned int *v52; // [rsp+38h] [rbp-48h]
  int v53; // [rsp+48h] [rbp-38h] BYREF
  int v54; // [rsp+4Ch] [rbp-34h] BYREF

  v6 = *a2;
  v49 = a1;
  v7 = *(_QWORD *)(a1 + 64) + 24 * v6;
  v8 = *(__int64 **)(v7 + 8);
  if ( !v8 )
    goto LABEL_4;
  v9 = *((unsigned int *)v8 + 3);
  v10 = (_DWORD *)v8[12];
  if ( (unsigned int)v9 > 1 )
  {
    if ( !sub_1369030(v10, &v10[v9], (_DWORD *)v7) )
      goto LABEL_4;
  }
  else if ( *(_DWORD *)v7 != *v10 )
  {
LABEL_4:
    v11 = (unsigned __int64 *)(v7 + 16);
    goto LABEL_5;
  }
  if ( !*((_BYTE *)v8 + 8) )
    goto LABEL_4;
  v41 = *v8;
  if ( !*v8
    || (v42 = *(unsigned int *)(v41 + 12), (unsigned int)v42 <= 1)
    || !sub_1369030(*(_DWORD **)(v41 + 96), (_DWORD *)(*(_QWORD *)(v41 + 96) + 4 * v42), (_DWORD *)v7)
    || (v11 = (unsigned __int64 *)(v41 + 152), !*(_BYTE *)(v41 + 8)) )
  {
    v11 = (unsigned __int64 *)(v8 + 19);
  }
LABEL_5:
  v12 = *v11;
  sub_1372DF0(a4);
  result = *(unsigned int **)a4;
  v14 = *(_DWORD *)(a4 + 80);
  v15 = *(unsigned int *)(a4 + 8);
  v50 = &v54;
  v51 = &v53;
  v16 = &result[4 * v15];
  v17 = result + 1;
  v52 = v16;
  if ( v16 != result )
  {
    while ( 1 )
    {
      v21 = *(_QWORD *)(v17 + 1);
      v22 = v14;
      v14 -= v21;
      sub_16AF710(v50, (unsigned int)v21, v22);
      v53 = v54;
      v23 = sub_16AF780(v51, v12);
      v24 = v12 < v23;
      v12 -= v23;
      v40 = v23;
      if ( v24 )
        v12 = 0;
      v25 = *(v17 - 1);
      if ( !v25 )
        break;
      if ( v25 == 2 )
      {
        v33 = *(unsigned int *)(a3 + 12);
        v34 = 0;
        if ( (unsigned int)v33 > 1 )
        {
          v35 = *(_DWORD **)(a3 + 96);
          v48 = v40;
          v36 = sub_13706F0(v35, (__int64)&v35[v33], v17);
          v40 = v48;
          v34 = 8 * (((__int64)v36 - v37) >> 2);
        }
        v32 = (__int64 *)(*(_QWORD *)(a3 + 128) + v34);
LABEL_25:
        v24 = __CFADD__(*v32, v40);
        v38 = *v32 + v40;
        if ( v24 )
          v38 = -1;
        result = v17 + 4;
        *v32 = v38;
        if ( v52 == v17 + 3 )
          return result;
        goto LABEL_11;
      }
      v18 = _mm_cvtsi32_si128(*v17).m128i_u64[0];
      v19 = *(unsigned int *)(a3 + 24);
      if ( (unsigned int)v19 >= *(_DWORD *)(a3 + 28) )
      {
        v48 = v40;
        v47 = v18;
        sub_16CD150(a3 + 16, a3 + 32, 0, 16);
        v19 = *(unsigned int *)(a3 + 24);
        v18 = v47;
        v40 = v48;
      }
      v20 = (unsigned __int64 *)(*(_QWORD *)(a3 + 16) + 16 * v19);
      v20[1] = v40;
      *v20 = v18;
      result = v17 + 4;
      ++*(_DWORD *)(a3 + 24);
      if ( v52 == v17 + 3 )
        return result;
LABEL_11:
      v17 = result;
    }
    v26 = *(_QWORD *)(v49 + 64) + 24LL * *v17;
    v27 = *(_QWORD *)(v26 + 8);
    if ( v27 )
    {
      v28 = *(unsigned int *)(v27 + 12);
      v29 = *(_DWORD **)(v27 + 96);
      if ( (unsigned int)v28 > 1 )
      {
        v47 = v40;
        v48 = v26;
        v45 = v27;
        v39 = sub_1369030(v29, &v29[v28], (_DWORD *)v26);
        v26 = v48;
        v40 = v47;
        if ( v39 )
        {
          v27 = v45;
          if ( *(_BYTE *)(v45 + 8) )
          {
LABEL_19:
            v30 = *(_QWORD *)v27;
            if ( !*(_QWORD *)v27 )
              goto LABEL_21;
            v31 = *(unsigned int *)(v30 + 12);
            if ( (unsigned int)v31 <= 1 )
              goto LABEL_21;
            v43 = *(_DWORD **)(v30 + 96);
            v46 = v27;
            v47 = v40;
            v48 = v30;
            v44 = sub_1369030(v43, &v43[v31], (_DWORD *)v26);
            v40 = v47;
            v27 = v46;
            if ( v44 )
            {
              v32 = (__int64 *)(v48 + 152);
              if ( !*(_BYTE *)(v48 + 8) )
                v32 = (__int64 *)(v46 + 152);
            }
            else
            {
LABEL_21:
              v32 = (__int64 *)(v27 + 152);
            }
            goto LABEL_25;
          }
        }
      }
      else if ( *(_DWORD *)v26 == *v29 && *(_BYTE *)(v27 + 8) )
      {
        goto LABEL_19;
      }
    }
    v32 = (__int64 *)(v26 + 16);
    goto LABEL_25;
  }
  return result;
}
