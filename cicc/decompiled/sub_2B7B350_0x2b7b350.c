// Function: sub_2B7B350
// Address: 0x2b7b350
//
__int64 __fastcall sub_2B7B350(__int64 *a1, __int64 a2, __int64 a3, int *a4, __int64 a5)
{
  __int64 v5; // r14
  __int64 v6; // r13
  __int64 v9; // rax
  __int64 v10; // r9
  __int64 v11; // rcx
  __int64 v12; // rax
  unsigned int **v13; // r15
  unsigned __int64 v14; // rax
  char v15; // al
  __int64 v16; // rdi
  __int64 v17; // rax
  __int64 v18; // rax
  __int64 v19; // r9
  __int64 v20; // rcx
  __int64 v21; // rax
  unsigned __int64 v22; // rax
  unsigned int **v23; // r15
  char v24; // al
  __int64 v25; // rdi
  __int64 v26; // rax
  __int64 v27; // r11
  __int64 v28; // rcx
  __int64 *v29; // r8
  __int64 v30; // rax
  __int64 v31; // r9
  __int64 v32; // r15
  __int64 v33; // rdx
  __int64 v34; // rsi
  __int64 v35; // rdi
  __int64 v36; // rdx
  __int64 v37; // r9
  __int64 ***v38; // rax
  __int64 v39; // r8
  __int64 v40; // r9
  unsigned int v41; // esi
  __int64 ***v42; // r13
  unsigned int v43; // ecx
  unsigned int v44; // eax
  int v45; // ecx
  __int64 v46; // rdx
  int v47; // esi
  __int64 result; // rax
  int *v49; // r11
  __int64 v50; // rcx
  __int64 ***v51; // rax
  __int64 v52; // rcx
  __int64 v53; // rsi
  __int64 v54; // rax
  _DWORD *v55; // rdx
  unsigned __int64 v56; // rcx
  __int64 v57; // rax
  __int64 v58; // rdi
  unsigned __int64 v59; // rdx
  signed __int64 v60; // r14
  unsigned __int64 v61; // r13
  __int64 v62; // r11
  int *v63; // r10
  __int64 v64; // rcx
  __int64 ***v65; // rax
  __int64 v66; // rcx
  __int64 v67; // rsi
  __int64 v68; // rax
  _DWORD *v69; // rdx
  __int64 v70; // [rsp+0h] [rbp-D0h]
  char v71; // [rsp+0h] [rbp-D0h]
  __int64 v72; // [rsp+0h] [rbp-D0h]
  char v73; // [rsp+0h] [rbp-D0h]
  __int64 v75; // [rsp+10h] [rbp-C0h]
  __int64 v76; // [rsp+18h] [rbp-B8h]
  _BYTE v77[32]; // [rsp+20h] [rbp-B0h] BYREF
  __int16 v78; // [rsp+40h] [rbp-90h]
  __m128i v79; // [rsp+50h] [rbp-80h] BYREF
  __int64 v80; // [rsp+60h] [rbp-70h]
  __int64 v81; // [rsp+68h] [rbp-68h]
  __int64 v82; // [rsp+70h] [rbp-60h]
  __int64 v83; // [rsp+78h] [rbp-58h]
  __int64 v84; // [rsp+80h] [rbp-50h]
  __int64 v85; // [rsp+88h] [rbp-48h]
  __int16 v86; // [rsp+90h] [rbp-40h]

  v5 = a3;
  v6 = a2;
  v9 = *a1;
  v10 = *(_QWORD *)(a2 + 8);
  v11 = *(_QWORD *)(v10 + 24);
  if ( (unsigned int)*(unsigned __int8 *)(*a1 + 8) - 17 > 1 )
  {
    if ( v11 == v9 )
    {
      v19 = *(_QWORD *)(a3 + 8);
      v20 = *(_QWORD *)(v19 + 24);
      goto LABEL_7;
    }
  }
  else if ( v11 == **(_QWORD **)(v9 + 16) )
  {
    v19 = *(_QWORD *)(a3 + 8);
    v20 = *(_QWORD *)(v19 + 24);
LABEL_6:
    v9 = **(_QWORD **)(v9 + 16);
    goto LABEL_7;
  }
  v12 = a1[15];
  v78 = 257;
  v13 = (unsigned int **)a1[14];
  v14 = *(_QWORD *)(v12 + 3344);
  v86 = 257;
  v70 = v10;
  v79 = (__m128i)v14;
  v80 = 0;
  v81 = 0;
  v82 = 0;
  v83 = 0;
  v84 = 0;
  v85 = 0;
  v15 = sub_9AC470(a2, &v79, 0);
  v16 = *a1;
  BYTE4(v76) = *(_BYTE *)(v70 + 8) == 18;
  LODWORD(v76) = *(_DWORD *)(v70 + 32);
  if ( (unsigned int)*(unsigned __int8 *)(*a1 + 8) - 17 <= 1 )
    v16 = **(_QWORD **)(v16 + 16);
  v71 = v15 ^ 1;
  v17 = sub_BCE1B0((__int64 *)v16, v76);
  v18 = sub_921630(v13, a2, v17, v71, (__int64)v77);
  v19 = *(_QWORD *)(v5 + 8);
  v6 = v18;
  v9 = *a1;
  v20 = *(_QWORD *)(v19 + 24);
  if ( (unsigned int)*(unsigned __int8 *)(*a1 + 8) - 17 <= 1 )
    goto LABEL_6;
LABEL_7:
  v72 = v19;
  if ( v9 != v20 )
  {
    v21 = a1[15];
    v78 = 257;
    v22 = *(_QWORD *)(v21 + 3344);
    v23 = (unsigned int **)a1[14];
    v86 = 257;
    v79 = (__m128i)v22;
    v80 = 0;
    v81 = 0;
    v82 = 0;
    v83 = 0;
    v84 = 0;
    v85 = 0;
    v24 = sub_9AC470(v5, &v79, 0);
    v25 = *a1;
    BYTE4(v75) = *(_BYTE *)(v72 + 8) == 18;
    LODWORD(v75) = *(_DWORD *)(v72 + 32);
    if ( (unsigned int)*(unsigned __int8 *)(*a1 + 8) - 17 <= 1 )
      v25 = **(_QWORD **)(v25 + 16);
    v73 = v24 ^ 1;
    v26 = sub_BCE1B0((__int64 *)v25, v75);
    v5 = sub_921630(v23, v5, v26, v73, (__int64)v77);
  }
  v27 = *((unsigned int *)a1 + 22);
  if ( (_DWORD)v27 )
  {
    v28 = a1[15];
    v29 = (__int64 *)a1[10];
    v30 = a1[14];
    v31 = *a1;
    v32 = *v29;
    v33 = *(_QWORD *)(v28 + 3344);
    v34 = v28 + 3160;
    v35 = v28 + 3112;
    if ( (_DWORD)v27 == 2 )
    {
      v62 = v29[1];
      v63 = (int *)a1[2];
      v79.m128i_i64[1] = v28 + 3112;
      v64 = *((unsigned int *)a1 + 6);
      v80 = v34;
      v81 = v33;
      v79.m128i_i64[0] = v30;
      v65 = sub_2B7A630(v32, v62, v63, v64, (__int64)&v79, v31);
      v66 = *((unsigned int *)a1 + 6);
      v67 = a1[2];
      v32 = (__int64)v65;
      v68 = 0;
      if ( (_DWORD)v66 )
      {
        do
        {
          v69 = (_DWORD *)(v67 + 4LL * (unsigned int)v68);
          if ( *v69 != -1 )
            *v69 = v68;
          ++v68;
        }
        while ( v66 != v68 );
        v30 = a1[14];
        v28 = a1[15];
        goto LABEL_14;
      }
    }
    else
    {
      if ( *(_DWORD *)(*(_QWORD *)(v32 + 8) + 32LL) == a5 )
      {
LABEL_14:
        v36 = *(_QWORD *)(v28 + 3344);
        v37 = *a1;
        v79.m128i_i64[0] = v30;
        v80 = v28 + 3160;
        v81 = v36;
        v79.m128i_i64[1] = v28 + 3112;
        v38 = sub_2B7A630(v6, v5, a4, a5, (__int64)&v79, v37);
        v41 = 1;
        v42 = v38;
        if ( *(_BYTE *)(*a1 + 8) == 17 )
          v41 = *(_DWORD *)(*a1 + 32);
        v43 = *((_DWORD *)v38[1] + 8) / v41;
        v44 = *(_DWORD *)(*(_QWORD *)(v32 + 8) + 32LL) / v41;
        if ( v43 >= v44 )
          v44 = v43;
        v45 = *((_DWORD *)a1 + 6);
        v46 = 0;
        v47 = v44 + v45;
        if ( v45 )
        {
          do
          {
            if ( a4[v46] != -1 )
              *(_DWORD *)(a1[2] + v46 * 4) = v44;
            ++v44;
            ++v46;
          }
          while ( v47 != v44 );
        }
        *(_QWORD *)a1[10] = v32;
        result = *((unsigned int *)a1 + 22);
        if ( (_DWORD)result == 2 )
        {
          result = a1[10];
          *(_QWORD *)(result + 8) = v42;
        }
        else
        {
          if ( result + 1 > (unsigned __int64)*((unsigned int *)a1 + 23) )
          {
            sub_C8D5F0((__int64)(a1 + 10), a1 + 12, result + 1, 8u, v39, v40);
            result = *((unsigned int *)a1 + 22);
          }
          *(_QWORD *)(a1[10] + 8 * result) = v42;
          ++*((_DWORD *)a1 + 22);
        }
        return result;
      }
      v49 = (int *)a1[2];
      v50 = *((unsigned int *)a1 + 6);
      v79.m128i_i64[1] = v35;
      v80 = v34;
      v81 = v33;
      v79.m128i_i64[0] = v30;
      v51 = sub_2B7A630(v32, 0, v49, v50, (__int64)&v79, v31);
      v52 = *((unsigned int *)a1 + 6);
      v53 = a1[2];
      v32 = (__int64)v51;
      v54 = 0;
      if ( (_DWORD)v52 )
      {
        do
        {
          v55 = (_DWORD *)(v53 + 4LL * (unsigned int)v54);
          if ( *v55 != -1 )
            *v55 = v54;
          ++v54;
        }
        while ( v52 != v54 );
      }
    }
    v30 = a1[14];
    v28 = a1[15];
    goto LABEL_14;
  }
  if ( !*((_DWORD *)a1 + 23) )
  {
    sub_C8D5F0((__int64)(a1 + 10), a1 + 12, 1u, 8u, a5, v19);
    v27 = *((unsigned int *)a1 + 22);
  }
  *(_QWORD *)(a1[10] + 8 * v27) = v6;
  v56 = *((unsigned int *)a1 + 23);
  v57 = (unsigned int)(*((_DWORD *)a1 + 22) + 1);
  *((_DWORD *)a1 + 22) = v57;
  if ( v57 + 1 > v56 )
  {
    sub_C8D5F0((__int64)(a1 + 10), a1 + 12, v57 + 1, 8u, a5, v19);
    v57 = *((unsigned int *)a1 + 22);
  }
  v58 = 0;
  *(_QWORD *)(a1[10] + 8 * v57) = v5;
  result = 0;
  v59 = *((unsigned int *)a1 + 7);
  ++*((_DWORD *)a1 + 22);
  v60 = 4 * a5;
  *((_DWORD *)a1 + 6) = 0;
  v61 = (4 * a5) >> 2;
  if ( v61 > v59 )
  {
    sub_C8D5F0((__int64)(a1 + 2), a1 + 4, v60 >> 2, 4u, a5, v19);
    result = *((unsigned int *)a1 + 6);
    v58 = 4 * result;
  }
  if ( v60 )
  {
    memcpy((void *)(a1[2] + v58), a4, v60);
    result = *((unsigned int *)a1 + 6);
  }
  *((_DWORD *)a1 + 6) = result + v61;
  return result;
}
