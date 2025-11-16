// Function: sub_2EA7D80
// Address: 0x2ea7d80
//
void __fastcall sub_2EA7D80(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rcx
  __int64 v8; // rax
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // r9
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // r8
  __int64 v19; // r9
  __int64 v20; // rsi
  unsigned int v21; // eax
  unsigned __int64 v22; // rdx
  __int64 v23; // rcx
  _QWORD *v24; // rsi
  __int64 v25; // r13
  __int64 v26; // rdi
  __int64 v27; // rsi
  __int64 v28; // r8
  __int64 v29; // r12
  _BYTE *v30; // rsi
  _QWORD *v31; // rax
  __int64 v32; // rax
  __int64 *v33; // rdx
  __int64 *v34; // rax
  __int64 *v35; // rdx
  __int64 v36; // rax
  __int64 *v37; // rax
  unsigned int v38; // r10d
  _QWORD v39[38]; // [rsp+30h] [rbp-750h] BYREF
  __int64 v40; // [rsp+160h] [rbp-620h] BYREF
  __int64 *v41; // [rsp+168h] [rbp-618h]
  int v42; // [rsp+170h] [rbp-610h]
  int v43; // [rsp+174h] [rbp-60Ch]
  int v44; // [rsp+178h] [rbp-608h]
  char v45; // [rsp+17Ch] [rbp-604h]
  __int64 v46; // [rsp+180h] [rbp-600h] BYREF
  __int64 *v47; // [rsp+1C0h] [rbp-5C0h]
  int v48; // [rsp+1C8h] [rbp-5B8h]
  int v49; // [rsp+1CCh] [rbp-5B4h]
  __int64 v50[24]; // [rsp+1D0h] [rbp-5B0h] BYREF
  char v51[8]; // [rsp+290h] [rbp-4F0h] BYREF
  unsigned __int64 v52; // [rsp+298h] [rbp-4E8h]
  char v53; // [rsp+2ACh] [rbp-4D4h]
  char v54[64]; // [rsp+2B0h] [rbp-4D0h] BYREF
  _BYTE *v55; // [rsp+2F0h] [rbp-490h] BYREF
  __int64 v56; // [rsp+2F8h] [rbp-488h]
  _BYTE v57[192]; // [rsp+300h] [rbp-480h] BYREF
  char v58[8]; // [rsp+3C0h] [rbp-3C0h] BYREF
  unsigned __int64 v59; // [rsp+3C8h] [rbp-3B8h]
  char v60; // [rsp+3DCh] [rbp-3A4h]
  char v61[64]; // [rsp+3E0h] [rbp-3A0h] BYREF
  _BYTE *v62; // [rsp+420h] [rbp-360h] BYREF
  __int64 v63; // [rsp+428h] [rbp-358h]
  _BYTE v64[192]; // [rsp+430h] [rbp-350h] BYREF
  char v65[8]; // [rsp+4F0h] [rbp-290h] BYREF
  unsigned __int64 v66; // [rsp+4F8h] [rbp-288h]
  char v67; // [rsp+50Ch] [rbp-274h]
  char *v68; // [rsp+550h] [rbp-230h] BYREF
  int v69; // [rsp+558h] [rbp-228h]
  char v70; // [rsp+560h] [rbp-220h] BYREF
  char v71[8]; // [rsp+620h] [rbp-160h] BYREF
  unsigned __int64 v72; // [rsp+628h] [rbp-158h]
  char v73; // [rsp+63Ch] [rbp-144h]
  char *v74; // [rsp+680h] [rbp-100h] BYREF
  unsigned int v75; // [rsp+688h] [rbp-F8h]
  char v76; // [rsp+690h] [rbp-F0h] BYREF

  memset(v39, 0, sizeof(v39));
  v7 = *(unsigned int *)(a2 + 120);
  v39[1] = &v39[4];
  v39[12] = &v39[14];
  v41 = &v46;
  v47 = v50;
  v8 = *(_QWORD *)(a2 + 112);
  v46 = a2;
  v50[1] = v8;
  v50[0] = v8 + 8 * v7;
  v50[2] = a2;
  LODWORD(v39[2]) = 8;
  BYTE4(v39[3]) = 1;
  HIDWORD(v39[13]) = 8;
  v42 = 8;
  v44 = 0;
  v45 = 1;
  v49 = 8;
  v43 = 1;
  v40 = 1;
  v48 = 1;
  sub_2EA7130((__int64)&v40, a2, a3, v50[0], a5, a6);
  sub_2EA7B20((__int64)v58, (__int64)v39);
  sub_2EA7B20((__int64)v51, (__int64)&v40);
  sub_2EA7B20((__int64)v65, (__int64)v51);
  sub_2EA7B20((__int64)v71, (__int64)v58);
  if ( v55 != v57 )
    _libc_free((unsigned __int64)v55);
  if ( !v53 )
    _libc_free(v52);
  if ( v62 != v64 )
    _libc_free((unsigned __int64)v62);
  if ( !v60 )
    _libc_free(v59);
  if ( v47 != v50 )
    _libc_free((unsigned __int64)v47);
  if ( !v45 )
    _libc_free((unsigned __int64)v41);
  if ( (_QWORD *)v39[12] != &v39[14] )
    _libc_free(v39[12]);
  if ( !BYTE4(v39[3]) )
    _libc_free(v39[1]);
  sub_C8CD80((__int64)v51, (__int64)v54, (__int64)v65, v9, v10, v11);
  v55 = v57;
  v56 = 0x800000000LL;
  if ( v69 )
    sub_2EA7C40((__int64)&v55, (__int64 *)&v68, v12, v13, v14, v15);
  sub_C8CD80((__int64)v58, (__int64)v61, (__int64)v71, v13, v14, v15);
  v20 = v75;
  v62 = v64;
  v63 = 0x800000000LL;
  if ( v75 )
  {
    sub_2EA7C40((__int64)&v62, (__int64 *)&v74, v16, v17, v18, v19);
    v20 = (unsigned int)v63;
  }
LABEL_21:
  v21 = v56;
  while ( 1 )
  {
    v22 = v21;
    v23 = 24LL * v21;
    if ( v21 != v20 )
      goto LABEL_26;
    v19 = (__int64)&v55[v23];
    if ( &v55[v23] == v55 )
      break;
    v24 = v62;
    v22 = (unsigned __int64)v55;
    while ( *(_QWORD *)(v22 + 16) == v24[2] && *(_QWORD *)(v22 + 8) == v24[1] && *(_QWORD *)v22 == *v24 )
    {
      v22 += 24LL;
      v24 += 3;
      if ( v19 == v22 )
        goto LABEL_51;
    }
LABEL_26:
    v25 = *(_QWORD *)&v55[v23 - 8];
    v26 = *a1;
    v27 = *(unsigned int *)(*a1 + 24);
    v28 = *(_QWORD *)(*a1 + 8);
    if ( !(_DWORD)v27 )
      goto LABEL_41;
    v27 = (unsigned int)(v27 - 1);
    v23 = (unsigned int)v27 & (((unsigned int)v25 >> 9) ^ ((unsigned int)v25 >> 4));
    v22 = v28 + 16 * v23;
    v19 = *(_QWORD *)v22;
    if ( v25 == *(_QWORD *)v22 )
    {
LABEL_28:
      v29 = *(_QWORD *)(v22 + 8);
      v39[0] = v29;
      if ( !v29 )
        goto LABEL_41;
      if ( v25 == **(_QWORD **)(v29 + 32) )
      {
        v32 = *(_QWORD *)v29;
        if ( *(_QWORD *)v29 )
        {
          v27 = *(_QWORD *)(v32 + 16);
          if ( v27 == *(_QWORD *)(v32 + 24) )
          {
            sub_2EA6DC0(v32 + 8, (_BYTE *)v27, v39);
            v29 = v39[0];
          }
          else
          {
            if ( v27 )
            {
              *(_QWORD *)v27 = v29;
              v27 = *(_QWORD *)(v32 + 16);
            }
            v27 += 8;
            *(_QWORD *)(v32 + 16) = v27;
          }
        }
        else
        {
          v40 = v29;
          v27 = *(_QWORD *)(v26 + 40);
          if ( v27 == *(_QWORD *)(v26 + 48) )
          {
            sub_2EA6DC0(v26 + 32, (_BYTE *)v27, &v40);
            v29 = v39[0];
          }
          else
          {
            if ( v27 )
            {
              *(_QWORD *)v27 = v29;
              v27 = *(_QWORD *)(v26 + 40);
              v29 = v39[0];
            }
            v27 += 8;
            *(_QWORD *)(v26 + 40) = v27;
          }
        }
        v33 = *(__int64 **)(v29 + 40);
        v34 = (__int64 *)(*(_QWORD *)(v29 + 32) + 8LL);
        if ( v33 != v34 )
        {
          v35 = v33 - 1;
          if ( v34 < v35 )
          {
            do
            {
              v23 = *v34;
              v27 = *v35;
              ++v34;
              --v35;
              *(v34 - 1) = v27;
              v35[1] = v23;
            }
            while ( v35 > v34 );
            v29 = v39[0];
          }
        }
        v36 = *(_QWORD *)(v29 + 16);
        v22 = *(_QWORD *)(v29 + 8);
        if ( v36 != v22 )
        {
          v37 = (__int64 *)(v36 - 8);
          if ( v22 < (unsigned __int64)v37 )
          {
            do
            {
              v23 = *(_QWORD *)v22;
              v27 = *v37;
              v22 += 8LL;
              --v37;
              *(_QWORD *)(v22 - 8) = v27;
              v37[1] = v23;
            }
            while ( v22 < (unsigned __int64)v37 );
            v29 = v39[0];
          }
        }
        v29 = *(_QWORD *)v29;
        v39[0] = v29;
        if ( !v29 )
        {
LABEL_40:
          v21 = v56;
          goto LABEL_41;
        }
      }
      while ( 2 )
      {
        v40 = v25;
        v30 = *(_BYTE **)(v29 + 40);
        if ( v30 == *(_BYTE **)(v29 + 48) )
        {
          sub_2E33A40(v29 + 32, v30, &v40);
          v27 = v40;
        }
        else
        {
          if ( v30 )
          {
            *(_QWORD *)v30 = v25;
            v30 = *(_BYTE **)(v29 + 40);
          }
          *(_QWORD *)(v29 + 40) = v30 + 8;
          v27 = v25;
        }
        if ( *(_BYTE *)(v29 + 84) )
        {
          v31 = *(_QWORD **)(v29 + 64);
          v23 = *(unsigned int *)(v29 + 76);
          v22 = (unsigned __int64)&v31[v23];
          if ( v31 != (_QWORD *)v22 )
          {
            while ( v27 != *v31 )
            {
              if ( (_QWORD *)v22 == ++v31 )
                goto LABEL_44;
            }
LABEL_39:
            v29 = *(_QWORD *)v39[0];
            v39[0] = v29;
            if ( !v29 )
              goto LABEL_40;
            continue;
          }
LABEL_44:
          if ( (unsigned int)v23 < *(_DWORD *)(v29 + 72) )
          {
            v23 = (unsigned int)(v23 + 1);
            *(_DWORD *)(v29 + 76) = v23;
            *(_QWORD *)v22 = v27;
            ++*(_QWORD *)(v29 + 56);
            goto LABEL_39;
          }
        }
        break;
      }
      sub_C8CC70(v29 + 56, v27, v22, v23, v28, v19);
      goto LABEL_39;
    }
    v22 = 1;
    while ( v19 != -4096 )
    {
      v38 = v22 + 1;
      v23 = (unsigned int)v27 & ((_DWORD)v23 + (_DWORD)v22);
      v22 = v28 + 16LL * (unsigned int)v23;
      v19 = *(_QWORD *)v22;
      if ( v25 == *(_QWORD *)v22 )
        goto LABEL_28;
      v22 = v38;
    }
LABEL_41:
    LODWORD(v56) = --v21;
    if ( v21 )
    {
      sub_2EA7130((__int64)v51, v27, v22, v23, v28, v19);
      v20 = (unsigned int)v63;
      goto LABEL_21;
    }
    v20 = (unsigned int)v63;
  }
LABEL_51:
  if ( v62 != v64 )
    _libc_free((unsigned __int64)v62);
  if ( !v60 )
    _libc_free(v59);
  if ( v55 != v57 )
    _libc_free((unsigned __int64)v55);
  if ( !v53 )
    _libc_free(v52);
  if ( v74 != &v76 )
    _libc_free((unsigned __int64)v74);
  if ( !v73 )
    _libc_free(v72);
  if ( v68 != &v70 )
    _libc_free((unsigned __int64)v68);
  if ( !v67 )
    _libc_free(v66);
}
