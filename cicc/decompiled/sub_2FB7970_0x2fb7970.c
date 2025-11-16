// Function: sub_2FB7970
// Address: 0x2fb7970
//
unsigned __int64 __fastcall sub_2FB7970(__int64 a1, int a2, int *a3, __int64 a4, unsigned __int8 a5, __int64 a6)
{
  int v8; // r14d
  __int64 v10; // r8
  unsigned __int64 v11; // rcx
  int v12; // r11d
  unsigned int v13; // eax
  __int64 v14; // rbx
  __int64 v15; // r12
  __int64 v16; // rax
  int v17; // r11d
  unsigned __int64 v18; // rbx
  unsigned __int64 v19; // r8
  __int64 v20; // rax
  __int64 v21; // rdi
  __int64 v22; // rcx
  unsigned int v23; // esi
  unsigned __int64 v24; // rbx
  int v25; // eax
  __int64 v26; // r10
  unsigned __int64 v27; // r9
  unsigned int i; // edx
  int *v29; // rbx
  int v30; // r15d
  unsigned int v31; // edx
  unsigned int v32; // eax
  __int64 v33; // rsi
  __int64 v34; // rax
  unsigned int v35; // r15d
  unsigned __int64 v37; // rdx
  __int64 v38; // rcx
  __int64 v39; // rax
  __int64 v40; // r12
  unsigned __int64 v41; // rax
  _QWORD *v42; // rdx
  _QWORD *v43; // rdi
  int v44; // eax
  int v45; // edx
  unsigned __int64 v46; // [rsp+8h] [rbp-78h]
  unsigned __int64 v47; // [rsp+8h] [rbp-78h]
  __int64 v48; // [rsp+8h] [rbp-78h]
  __int64 v49; // [rsp+10h] [rbp-70h]
  __int64 v50; // [rsp+10h] [rbp-70h]
  unsigned __int64 v51; // [rsp+10h] [rbp-70h]
  int *v52; // [rsp+18h] [rbp-68h]
  __int64 v53; // [rsp+18h] [rbp-68h]
  __int64 v54; // [rsp+18h] [rbp-68h]
  __int64 v55; // [rsp+18h] [rbp-68h]
  __int64 v56; // [rsp+18h] [rbp-68h]
  int v57; // [rsp+18h] [rbp-68h]
  int v58; // [rsp+20h] [rbp-60h]
  _QWORD *v59; // [rsp+20h] [rbp-60h]
  unsigned __int64 v60; // [rsp+20h] [rbp-60h]
  int v61; // [rsp+20h] [rbp-60h]
  unsigned __int64 v62; // [rsp+20h] [rbp-60h]
  __int64 v63; // [rsp+20h] [rbp-60h]
  unsigned __int64 v65; // [rsp+28h] [rbp-58h]
  __int64 v66; // [rsp+28h] [rbp-58h]
  int *v67; // [rsp+38h] [rbp-48h] BYREF
  int v68; // [rsp+40h] [rbp-40h] BYREF
  int v69; // [rsp+44h] [rbp-3Ch]
  unsigned __int64 v70; // [rsp+48h] [rbp-38h]

  v8 = a2;
  v10 = *(_QWORD *)(a1 + 8);
  v11 = *(unsigned int *)(v10 + 160);
  v12 = *(_DWORD *)(**(_QWORD **)(*(_QWORD *)(a1 + 72) + 16LL)
                  + 4LL * (unsigned int)(*(_DWORD *)(*(_QWORD *)(a1 + 72) + 64LL) + a2));
  v13 = v12 & 0x7FFFFFFF;
  v14 = 8LL * (v12 & 0x7FFFFFFF);
  if ( (v12 & 0x7FFFFFFFu) >= (unsigned int)v11 || (v15 = *(_QWORD *)(*(_QWORD *)(v10 + 152) + 8LL * v13)) == 0 )
  {
    v32 = v13 + 1;
    if ( (unsigned int)v11 < v32 )
    {
      v37 = v32;
      if ( v32 != v11 )
      {
        if ( v32 >= v11 )
        {
          v40 = *(_QWORD *)(v10 + 168);
          v41 = v32 - v11;
          if ( v37 > *(unsigned int *)(v10 + 164) )
          {
            v48 = a4;
            v51 = v41;
            v57 = *(_DWORD *)(**(_QWORD **)(*(_QWORD *)(a1 + 72) + 16LL)
                            + 4LL * (unsigned int)(*(_DWORD *)(*(_QWORD *)(a1 + 72) + 64LL) + a2));
            v63 = *(_QWORD *)(a1 + 8);
            sub_C8D5F0(v10 + 152, (const void *)(v10 + 168), v37, 8u, v10, a6);
            v10 = v63;
            a4 = v48;
            v41 = v51;
            v12 = v57;
            v11 = *(unsigned int *)(v63 + 160);
          }
          v33 = *(_QWORD *)(v10 + 152);
          v42 = (_QWORD *)(v33 + 8 * v11);
          v43 = &v42[v41];
          if ( v42 != v43 )
          {
            do
              *v42++ = v40;
            while ( v43 != v42 );
            LODWORD(v11) = *(_DWORD *)(v10 + 160);
            v33 = *(_QWORD *)(v10 + 152);
          }
          *(_DWORD *)(v10 + 160) = v41 + v11;
          goto LABEL_23;
        }
        *(_DWORD *)(v10 + 160) = v32;
      }
    }
    v33 = *(_QWORD *)(v10 + 152);
LABEL_23:
    v53 = a4;
    v59 = (_QWORD *)v10;
    v34 = sub_2E10F30(v12);
    *(_QWORD *)(v33 + v14) = v34;
    v15 = v34;
    sub_2E11E80(v59, v34);
    v10 = *(_QWORD *)(a1 + 8);
    a4 = v53;
  }
  v16 = *(_QWORD *)(v10 + 56);
  v17 = *(_DWORD *)(v15 + 72);
  *(_QWORD *)(v10 + 136) += 16LL;
  v18 = (v16 + 15) & 0xFFFFFFFFFFFFFFF0LL;
  if ( *(_QWORD *)(v10 + 64) >= v18 + 16 && v16 )
  {
    *(_QWORD *)(v10 + 56) = v18 + 16;
    v19 = (v16 + 15) & 0xFFFFFFFFFFFFFFF0LL;
    if ( !v18 )
      goto LABEL_7;
  }
  else
  {
    v54 = a4;
    v61 = v17;
    v39 = sub_9D1E70(v10 + 56, 16, 16, 4);
    a4 = v54;
    v17 = v61;
    v19 = v39;
    v18 = v39;
  }
  *(_DWORD *)v19 = v17;
  *(_QWORD *)(v19 + 8) = a4;
LABEL_7:
  v20 = *(unsigned int *)(v15 + 72);
  if ( v20 + 1 > (unsigned __int64)*(unsigned int *)(v15 + 76) )
  {
    v62 = v19;
    sub_C8D5F0(v15 + 64, (const void *)(v15 + 80), v20 + 1, 8u, v19, a6);
    v20 = *(unsigned int *)(v15 + 72);
    v19 = v62;
  }
  v21 = a1 + 392;
  *(_QWORD *)(*(_QWORD *)(v15 + 64) + 8 * v20) = v18;
  v22 = *(_QWORD *)(v15 + 104);
  ++*(_DWORD *)(v15 + 72);
  v23 = *(_DWORD *)(a1 + 416);
  v68 = v8;
  v24 = v18 & 0xFFFFFFFFFFFFFFFBLL;
  if ( v22 )
    v24 = 0;
  v25 = *a3;
  v26 = 4LL * (v22 != 0);
  v69 = *a3;
  v70 = v26 | v24;
  if ( v23 )
  {
    v58 = 1;
    v52 = 0;
    v27 = v23 - 1;
    for ( i = v27
            & (((0xBF58476D1CE4E5B9LL * ((unsigned int)(37 * v25) | ((unsigned __int64)(unsigned int)(37 * v8) << 32))) >> 31)
             ^ (756364221 * v25)); ; i = v27 & v31 )
    {
      v29 = (int *)(*(_QWORD *)(a1 + 400) + 16LL * i);
      v30 = *v29;
      if ( v8 == *v29 && v25 == v29[1] )
        break;
      if ( v30 == -1 )
      {
        if ( v29[1] == -1 )
        {
          if ( v52 )
            v29 = v52;
          v44 = *(_DWORD *)(a1 + 408);
          ++*(_QWORD *)(a1 + 392);
          v45 = v44 + 1;
          v67 = v29;
          if ( 4 * (v44 + 1) < 3 * v23 )
          {
            v27 = v23 >> 3;
            if ( v23 - *(_DWORD *)(a1 + 412) - v45 <= (unsigned int)v27 )
            {
              v47 = v19;
              v50 = 4LL * (v22 != 0);
              v56 = v22;
              sub_2FB76D0(v21, v23);
              sub_2FB3720(v21, &v68, &v67);
              v8 = v68;
              v29 = v67;
              v19 = v47;
              v26 = v50;
              v22 = v56;
              v45 = *(_DWORD *)(a1 + 408) + 1;
            }
            goto LABEL_45;
          }
          goto LABEL_52;
        }
      }
      else if ( v30 == -2 && v29[1] == -2 )
      {
        if ( v52 )
          v29 = v52;
        v52 = v29;
      }
      v31 = v58 + i;
      ++v58;
    }
    goto LABEL_25;
  }
  ++*(_QWORD *)(a1 + 392);
  v67 = 0;
LABEL_52:
  v46 = v19;
  v49 = 4LL * (v22 != 0);
  v55 = v22;
  sub_2FB76D0(v21, 2 * v23);
  sub_2FB3720(a1 + 392, &v68, &v67);
  v8 = v68;
  v29 = v67;
  v22 = v55;
  v26 = v49;
  v19 = v46;
  v45 = *(_DWORD *)(a1 + 408) + 1;
LABEL_45:
  *(_DWORD *)(a1 + 408) = v45;
  if ( *v29 != -1 || v29[1] != -1 )
    --*(_DWORD *)(a1 + 412);
  *v29 = v8;
  v29[1] = v69;
  *((_QWORD *)v29 + 1) = v70;
  if ( v22 )
  {
LABEL_25:
    v35 = a5;
    if ( (*((_QWORD *)v29 + 1) & 0xFFFFFFFFFFFFFFF8LL) != 0 )
    {
      v38 = a5;
      v60 = v19;
      v66 = v26;
      sub_2FB21F0((_QWORD *)a1, v15, *((_QWORD *)v29 + 1) & 0xFFFFFFFFFFFFFFF8LL, v38, v19, v27);
      v19 = v60;
      *((_QWORD *)v29 + 1) = v66;
    }
    v65 = v19;
    sub_2FB21F0((_QWORD *)a1, v15, v19, v35, v19, v27);
    return v65;
  }
  return v19;
}
