// Function: sub_34F5900
// Address: 0x34f5900
//
void __fastcall sub_34F5900(__int64 a1, __int64 a2, __int64 a3, unsigned __int64 a4, __int64 a5, __int64 a6)
{
  _QWORD *v7; // rdi
  unsigned int v8; // eax
  bool v9; // zf
  __int64 *v10; // rdx
  __int64 v11; // r14
  __int64 v12; // rbx
  __int64 *v13; // rax
  char v14; // dl
  __int64 v15; // rax
  unsigned __int64 v16; // rax
  __int64 v17; // r14
  _QWORD *v18; // rax
  _QWORD *v19; // rdx
  __int64 v20; // r9
  int v21; // r10d
  unsigned __int64 v22; // rdx
  unsigned int v23; // eax
  __int64 v24; // r14
  __int64 v25; // r15
  unsigned __int64 v26; // rbx
  __int64 *v27; // rdx
  __int64 v28; // rbx
  __int64 v29; // rax
  unsigned __int64 v30; // rdx
  __int64 *v31; // rax
  __int64 v32; // r11
  __int64 v33; // rdx
  __int64 v34; // rax
  __int64 v35; // r15
  signed __int64 v36; // rbx
  __int64 *v37; // rdx
  __int64 v38; // rbx
  __int64 v39; // rax
  unsigned __int64 v40; // rdx
  __int64 *v41; // rax
  __int64 v42; // rdx
  unsigned int v43; // eax
  __int64 v44; // rsi
  __int64 v45; // rax
  unsigned __int64 v46; // r11
  __int64 v47; // r15
  unsigned __int64 v48; // rax
  _QWORD *v49; // rcx
  _QWORD *v50; // rdi
  __int64 v51; // rsi
  _QWORD *v52; // rdi
  unsigned __int64 v53; // [rsp+0h] [rbp-F0h]
  int v54; // [rsp+8h] [rbp-E8h]
  __int64 v55; // [rsp+18h] [rbp-D8h]
  _QWORD *v56; // [rsp+18h] [rbp-D8h]
  __int64 v57; // [rsp+18h] [rbp-D8h]
  __int64 v58; // [rsp+28h] [rbp-C8h] BYREF
  _QWORD *v59; // [rsp+30h] [rbp-C0h] BYREF
  __int64 v60; // [rsp+38h] [rbp-B8h]
  _QWORD v61[22]; // [rsp+40h] [rbp-B0h] BYREF

  v7 = v61;
  v59 = v61;
  v61[0] = a2;
  v61[1] = a3;
  v60 = 0x800000001LL;
  v8 = 1;
  do
  {
    v9 = *(_BYTE *)(a1 + 308) == 0;
    v10 = &v7[2 * v8 - 2];
    v11 = *v10;
    v12 = v10[1];
    LODWORD(v60) = v8 - 1;
    if ( v9 )
      goto LABEL_12;
    v13 = *(__int64 **)(a1 + 288);
    a4 = *(unsigned int *)(a1 + 300);
    v10 = &v13[a4];
    if ( v13 != v10 )
    {
      while ( v12 != *v13 )
      {
        if ( v10 == ++v13 )
          goto LABEL_28;
      }
      goto LABEL_7;
    }
LABEL_28:
    if ( (unsigned int)a4 >= *(_DWORD *)(a1 + 296) )
    {
LABEL_12:
      sub_C8CC70(a1 + 280, v12, (__int64)v10, a4, a5, a6);
      if ( !v14 )
        goto LABEL_7;
      v15 = *(_QWORD *)(v12 + 8);
      if ( (v15 & 6) == 0 )
      {
LABEL_30:
        v32 = *(_QWORD *)(*(_QWORD *)(a1 + 16) + 32LL);
        v33 = *(_QWORD *)((v15 & 0xFFFFFFFFFFFFFFF8LL) + 16);
        if ( v33 )
        {
          v34 = *(_QWORD *)(v33 + 24);
        }
        else
        {
          v51 = *(unsigned int *)(v32 + 304);
          v52 = *(_QWORD **)(v32 + 296);
          v58 = v15;
          v34 = *(sub_34F5130(v52, (__int64)&v52[2 * v51], &v58) - 1);
        }
        v35 = *(_QWORD *)(v34 + 64);
        v55 = v35 + 8LL * *(unsigned int *)(v34 + 72);
        if ( v35 != v55 )
        {
          while ( 1 )
          {
            v42 = *(_QWORD *)(*(_QWORD *)(v32 + 152) + 16LL * *(unsigned int *)(*(_QWORD *)v35 + 24LL) + 8);
            if ( ((v42 >> 1) & 3) != 0 )
              v36 = v42 & 0xFFFFFFFFFFFFFFF8LL | (2LL * (int)(((v42 >> 1) & 3) - 1));
            else
              v36 = *(_QWORD *)(v42 & 0xFFFFFFFFFFFFFFF8LL) & 0xFFFFFFFFFFFFFFF8LL | 6;
            v37 = (__int64 *)sub_2E09D00((__int64 *)v11, v36);
            a4 = 3LL * *(unsigned int *)(v11 + 8);
            if ( v37 != (__int64 *)(*(_QWORD *)v11 + 24LL * *(unsigned int *)(v11 + 8)) )
            {
              a4 = v36 & 0xFFFFFFFFFFFFFFF8LL;
              if ( (*(_DWORD *)((*v37 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v37 >> 1) & 3) <= (*(_DWORD *)((v36 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v36 >> 1) & 3) )
              {
                v38 = v37[2];
                if ( v38 )
                {
                  v39 = (unsigned int)v60;
                  a4 = HIDWORD(v60);
                  v40 = (unsigned int)v60 + 1LL;
                  if ( v40 > HIDWORD(v60) )
                  {
                    sub_C8D5F0((__int64)&v59, v61, v40, 0x10u, a5, a6);
                    v39 = (unsigned int)v60;
                  }
                  v41 = &v59[2 * v39];
                  *v41 = v11;
                  v41[1] = v38;
                  LODWORD(v60) = v60 + 1;
                }
              }
            }
            v35 += 8;
            if ( v55 == v35 )
              break;
            v32 = *(_QWORD *)(*(_QWORD *)(a1 + 16) + 32LL);
          }
        }
LABEL_7:
        v8 = v60;
        goto LABEL_8;
      }
    }
    else
    {
      a4 = (unsigned int)(a4 + 1);
      *(_DWORD *)(a1 + 300) = a4;
      *v10 = v12;
      ++*(_QWORD *)(a1 + 280);
      v15 = *(_QWORD *)(v12 + 8);
      if ( (v15 & 6) == 0 )
        goto LABEL_30;
    }
    v16 = v15 & 0xFFFFFFFFFFFFFFF8LL;
    v17 = *(_QWORD *)(v16 + 16);
    if ( *(_BYTE *)(a1 + 212) )
    {
      v18 = *(_QWORD **)(a1 + 192);
      v19 = &v18[*(unsigned int *)(a1 + 204)];
      if ( v18 == v19 )
        goto LABEL_7;
      while ( v17 != *v18 )
      {
        if ( v19 == ++v18 )
          goto LABEL_7;
      }
    }
    else if ( !sub_C8CA60(a1 + 184, *(_QWORD *)(v16 + 16)) )
    {
      goto LABEL_7;
    }
    v20 = *(_QWORD *)(a1 + 16);
    v21 = *(_DWORD *)(*(_QWORD *)(v17 + 32) + 48LL);
    v22 = *(unsigned int *)(v20 + 160);
    v23 = v21 & 0x7FFFFFFF;
    v24 = 8LL * (v21 & 0x7FFFFFFF);
    if ( (v21 & 0x7FFFFFFFu) >= (unsigned int)v22 || (v25 = *(_QWORD *)(*(_QWORD *)(v20 + 152) + 8LL * v23)) == 0 )
    {
      v43 = v23 + 1;
      if ( (unsigned int)v22 < v43 )
      {
        v46 = v43;
        if ( v43 != v22 )
        {
          if ( v43 >= v22 )
          {
            v47 = *(_QWORD *)(v20 + 168);
            v48 = v43 - v22;
            if ( v46 > *(unsigned int *)(v20 + 164) )
            {
              v53 = v48;
              v54 = v21;
              v57 = *(_QWORD *)(a1 + 16);
              sub_C8D5F0(v20 + 152, (const void *)(v20 + 168), v46, 8u, a5, v20);
              v20 = v57;
              v48 = v53;
              v21 = v54;
              v22 = *(unsigned int *)(v57 + 160);
            }
            v44 = *(_QWORD *)(v20 + 152);
            v49 = (_QWORD *)(v44 + 8 * v22);
            v50 = &v49[v48];
            if ( v49 != v50 )
            {
              do
                *v49++ = v47;
              while ( v50 != v49 );
              LODWORD(v22) = *(_DWORD *)(v20 + 160);
              v44 = *(_QWORD *)(v20 + 152);
            }
            *(_DWORD *)(v20 + 160) = v48 + v22;
            goto LABEL_49;
          }
          *(_DWORD *)(v20 + 160) = v43;
        }
      }
      v44 = *(_QWORD *)(v20 + 152);
LABEL_49:
      v56 = (_QWORD *)v20;
      v45 = sub_2E10F30(v21);
      *(_QWORD *)(v44 + v24) = v45;
      v25 = v45;
      sub_2E11E80(v56, v45);
    }
    v26 = *(_QWORD *)(v12 + 8) & 0xFFFFFFFFFFFFFFF8LL;
    v27 = (__int64 *)sub_2E09D00((__int64 *)v25, v26 | 2);
    if ( v27 == (__int64 *)(*(_QWORD *)v25 + 24LL * *(unsigned int *)(v25 + 8))
      || (*(_DWORD *)((*v27 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v27 >> 1) & 3) > (*(_DWORD *)(v26 + 24) | 1u) )
    {
      v28 = 0;
    }
    else
    {
      v28 = v27[2];
    }
    v29 = (unsigned int)v60;
    a4 = HIDWORD(v60);
    v30 = (unsigned int)v60 + 1LL;
    if ( v30 > HIDWORD(v60) )
    {
      sub_C8D5F0((__int64)&v59, v61, v30, 0x10u, a5, a6);
      v29 = (unsigned int)v60;
    }
    v31 = &v59[2 * v29];
    *v31 = v25;
    v31[1] = v28;
    v8 = v60 + 1;
    LODWORD(v60) = v60 + 1;
LABEL_8:
    v7 = v59;
  }
  while ( v8 );
  if ( v59 != v61 )
    _libc_free((unsigned __int64)v59);
}
