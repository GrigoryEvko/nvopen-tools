// Function: sub_2895860
// Address: 0x2895860
//
__int64 __fastcall sub_2895860(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // r13
  __int64 v6; // rdi
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 v9; // r8
  unsigned int v10; // edx
  __int64 *v11; // rcx
  __int64 v12; // r9
  __int64 v13; // rdi
  __int64 v14; // rbx
  char *v15; // rsi
  int v16; // edx
  unsigned int v17; // eax
  __int64 v18; // rcx
  __int64 v19; // rdx
  __int64 v20; // r9
  __m128i v21; // xmm0
  unsigned int v23; // ebx
  char v24; // si
  int v25; // edx
  int v26; // ecx
  __int64 v27; // rax
  __int64 v28; // r9
  _BYTE *v29; // r10
  __int64 v30; // r12
  __int64 v31; // rdx
  unsigned __int64 v32; // r8
  unsigned int v33; // eax
  int v34; // edi
  char *v35; // r14
  __int64 v36; // r15
  __int64 v37; // rdi
  __int64 (__fastcall *v38)(__int64, __int64, _BYTE *, char *, __int64); // rax
  _QWORD *v39; // rax
  unsigned int *v40; // r15
  __int64 v41; // r14
  __int64 v42; // rdx
  unsigned int v43; // esi
  __int64 v44; // rax
  __int64 v45; // rbx
  void *v46; // r13
  __int64 v47; // r12
  void *v48; // rdi
  _BYTE *v49; // rdi
  int v50; // edx
  int v51; // ecx
  int v52; // r10d
  _BYTE *v54; // [rsp+20h] [rbp-1A0h]
  __int64 v55; // [rsp+20h] [rbp-1A0h]
  _BYTE *v56; // [rsp+20h] [rbp-1A0h]
  __int64 v57; // [rsp+38h] [rbp-188h]
  char *v60; // [rsp+50h] [rbp-170h] BYREF
  char v61; // [rsp+70h] [rbp-150h]
  char v62; // [rsp+71h] [rbp-14Fh]
  _BYTE v63[32]; // [rsp+80h] [rbp-140h] BYREF
  __int16 v64; // [rsp+A0h] [rbp-120h]
  char *v65; // [rsp+B0h] [rbp-110h] BYREF
  unsigned int v66; // [rsp+B8h] [rbp-108h]
  char v67; // [rsp+C0h] [rbp-100h] BYREF
  void *src; // [rsp+100h] [rbp-C0h] BYREF
  __int64 v69; // [rsp+108h] [rbp-B8h]
  _BYTE v70[176]; // [rsp+110h] [rbp-B0h] BYREF

  v5 = a3;
  v6 = *(_QWORD *)(a2 + 248);
  v7 = 0;
  if ( (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(a3 + 8) + 8LL) - 17 < 2 )
    v7 = *(_QWORD *)(a3 + 8);
  v57 = v7;
  v8 = *(unsigned int *)(a2 + 264);
  if ( (_DWORD)v8 )
  {
    v9 = (unsigned int)(v8 - 1);
    v10 = v9 & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
    v11 = (__int64 *)(v6 + 16LL * v10);
    v12 = *v11;
    if ( *v11 != v5 )
    {
      v51 = 1;
      while ( v12 != -4096 )
      {
        v52 = v51 + 1;
        v10 = v9 & (v51 + v10);
        v11 = (__int64 *)(v6 + 16LL * v10);
        v12 = *v11;
        if ( *v11 == v5 )
          goto LABEL_5;
        v51 = v52;
      }
      goto LABEL_18;
    }
LABEL_5:
    if ( v11 != (__int64 *)(v6 + 16 * v8) )
    {
      v13 = *(_QWORD *)(a2 + 272);
      v14 = v13 + 176LL * *((unsigned int *)v11 + 2);
      if ( v14 != v13 + 176LL * *(unsigned int *)(a2 + 280) )
      {
        v15 = *(char **)(v14 + 8);
        v16 = *(_DWORD *)a4;
        v17 = *(_DWORD *)(v14 + 16);
        if ( *(_BYTE *)(v14 + 168) )
        {
          if ( v16 == *(_DWORD *)(*(_QWORD *)(*(_QWORD *)v15 + 8LL) + 32LL) )
          {
            v18 = *(unsigned int *)(a4 + 4);
            v19 = v17;
            goto LABEL_10;
          }
        }
        else if ( v16 == v17 )
        {
          v18 = *(unsigned int *)(a4 + 4);
          v19 = *(unsigned int *)(*(_QWORD *)(*(_QWORD *)v15 + 8LL) + 32LL);
LABEL_10:
          if ( (_DWORD)v18 == (_DWORD)v19 )
          {
            *(_QWORD *)a1 = a1 + 16;
            *(_QWORD *)(a1 + 8) = 0x1000000000LL;
            v20 = *(unsigned int *)(v14 + 16);
            if ( (_DWORD)v20 )
              sub_2894AD0(a1, v14 + 8, v19, v18, v9, v20);
            v21 = _mm_loadu_si128((const __m128i *)(v14 + 152));
            *(_BYTE *)(a1 + 160) = *(_BYTE *)(v14 + 168);
            *(__m128i *)(a1 + 144) = v21;
            return a1;
          }
        }
        if ( v17 == 1 )
          v5 = *(_QWORD *)v15;
        else
          v5 = sub_9B9840((unsigned int **)a5, v15, v17);
      }
    }
  }
LABEL_18:
  src = v70;
  v69 = 0x1000000000LL;
  if ( !*(_DWORD *)(v57 + 32) )
  {
    LODWORD(v47) = 0;
    *(_DWORD *)(a1 + 12) = 16;
    *(_QWORD *)a1 = a1 + 16;
    LODWORD(v45) = 0;
    goto LABEL_42;
  }
  v23 = 0;
  v24 = *(_BYTE *)(a4 + 8);
  v25 = *(_DWORD *)a4;
  v26 = *(_DWORD *)(a4 + 4);
  do
  {
    if ( !v24 )
      v25 = v26;
    v60 = "split";
    v62 = 1;
    v61 = 3;
    sub_9B9680((__int64 *)&v65, v23, v25, 0);
    v35 = v65;
    v36 = v66;
    v29 = (_BYTE *)sub_ACADE0(*(__int64 ***)(v5 + 8));
    v37 = *(_QWORD *)(a5 + 80);
    v38 = *(__int64 (__fastcall **)(__int64, __int64, _BYTE *, char *, __int64))(*(_QWORD *)v37 + 112LL);
    if ( (char *)v38 != (char *)sub_9B6630 )
    {
      v56 = v29;
      v44 = v38(v37, v5, v29, v35, v36);
      v29 = v56;
      v30 = v44;
LABEL_22:
      if ( v30 )
        goto LABEL_23;
      goto LABEL_34;
    }
    if ( *(_BYTE *)v5 <= 0x15u && *v29 <= 0x15u )
    {
      v54 = v29;
      v27 = sub_AD5CE0(v5, (__int64)v29, v35, v36, 0);
      v29 = v54;
      v30 = v27;
      goto LABEL_22;
    }
LABEL_34:
    v55 = (__int64)v29;
    v64 = 257;
    v39 = sub_BD2C40(112, unk_3F1FE60);
    v30 = (__int64)v39;
    if ( v39 )
      sub_B4E9E0((__int64)v39, v5, v55, v35, v36, (__int64)v63, 0, 0);
    (*(void (__fastcall **)(_QWORD, __int64, char **, _QWORD, _QWORD))(**(_QWORD **)(a5 + 88) + 16LL))(
      *(_QWORD *)(a5 + 88),
      v30,
      &v60,
      *(_QWORD *)(a5 + 56),
      *(_QWORD *)(a5 + 64));
    v40 = *(unsigned int **)a5;
    v41 = *(_QWORD *)a5 + 16LL * *(unsigned int *)(a5 + 8);
    if ( *(_QWORD *)a5 != v41 )
    {
      do
      {
        v42 = *((_QWORD *)v40 + 1);
        v43 = *v40;
        v40 += 4;
        sub_B99FD0(v30, v43, v42);
      }
      while ( (unsigned int *)v41 != v40 );
    }
LABEL_23:
    if ( v65 != &v67 )
      _libc_free((unsigned __int64)v65);
    v31 = (unsigned int)v69;
    v32 = (unsigned int)v69 + 1LL;
    if ( v32 > HIDWORD(v69) )
    {
      sub_C8D5F0((__int64)&src, v70, (unsigned int)v69 + 1LL, 8u, v32, v28);
      v31 = (unsigned int)v69;
    }
    *((_QWORD *)src + v31) = v30;
    v24 = *(_BYTE *)(a4 + 8);
    v25 = *(_DWORD *)a4;
    v26 = *(_DWORD *)(a4 + 4);
    v33 = v69 + 1;
    v34 = *(_DWORD *)a4;
    LODWORD(v69) = v69 + 1;
    if ( !v24 )
      v34 = v26;
    v23 += v34;
  }
  while ( v23 < *(_DWORD *)(v57 + 32) );
  v45 = v33;
  v46 = src;
  v47 = 8LL * v33;
  v48 = (void *)(a1 + 16);
  *(_QWORD *)(a1 + 8) = 0x1000000000LL;
  *(_QWORD *)a1 = a1 + 16;
  if ( v33 > 0x10uLL )
  {
    sub_C8D5F0(a1, v48, v33, 8u, v32, v28);
    v48 = (void *)(*(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8));
    goto LABEL_45;
  }
  if ( v47 )
  {
LABEL_45:
    memcpy(v48, v46, 8 * v45);
    LODWORD(v47) = *(_DWORD *)(a1 + 8);
  }
LABEL_42:
  v49 = src;
  *(_DWORD *)(a1 + 8) = v47 + v45;
  v50 = dword_5003CC8;
  *(_QWORD *)(a1 + 144) = 0;
  *(_QWORD *)(a1 + 152) = 0;
  *(_BYTE *)(a1 + 160) = v50 == 0;
  if ( v49 != v70 )
    _libc_free((unsigned __int64)v49);
  return a1;
}
