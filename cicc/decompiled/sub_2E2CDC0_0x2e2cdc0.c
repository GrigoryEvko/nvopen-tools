// Function: sub_2E2CDC0
// Address: 0x2e2cdc0
//
__int64 __fastcall sub_2E2CDC0(_QWORD *a1, __int64 a2)
{
  __int64 v2; // rbx
  __int64 v3; // r14
  __int64 (*v4)(); // rax
  __int64 v5; // rax
  __int64 v6; // r8
  __int64 v7; // r11
  __int64 v8; // rcx
  __int64 v9; // r13
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 v12; // rbx
  __int64 v13; // r12
  void **p_base; // r10
  unsigned __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // rdx
  int v18; // r15d
  __int64 (*v19)(); // rax
  _BYTE *v20; // rdi
  __int64 v21; // r15
  unsigned int v22; // r9d
  _BYTE *v23; // rax
  _QWORD *v24; // rcx
  unsigned int v25; // r13d
  int v26; // edx
  __int64 v27; // r10
  __int64 v28; // rax
  int v29; // edx
  __int64 v30; // r11
  __int64 v31; // r8
  __int64 v32; // rbx
  __int64 v33; // rbx
  char v34; // al
  __int64 v35; // rbx
  __int64 (*v36)(); // rax
  char v37; // al
  __int64 v38; // r8
  __int64 v39; // r11
  __int64 v40; // rcx
  unsigned int v41; // eax
  char v43; // al
  __int64 v44; // rax
  char *v45; // rdx
  const __m128i *v46; // r15
  __m128i *v47; // rax
  __int64 v48; // rdx
  __int64 v49; // rax
  signed __int64 v50; // r15
  __int64 v51; // [rsp+8h] [rbp-6A8h]
  __int64 v52; // [rsp+8h] [rbp-6A8h]
  __int64 v53; // [rsp+10h] [rbp-6A0h]
  __int64 v54; // [rsp+10h] [rbp-6A0h]
  unsigned int v55; // [rsp+10h] [rbp-6A0h]
  __int64 v56; // [rsp+18h] [rbp-698h]
  __int64 v57; // [rsp+18h] [rbp-698h]
  __int64 v58; // [rsp+18h] [rbp-698h]
  __int64 v59; // [rsp+18h] [rbp-698h]
  unsigned int v60; // [rsp+20h] [rbp-690h]
  __int64 v61; // [rsp+20h] [rbp-690h]
  __int64 v62; // [rsp+28h] [rbp-688h]
  __int64 v63; // [rsp+30h] [rbp-680h]
  unsigned int v64; // [rsp+30h] [rbp-680h]
  unsigned int v65; // [rsp+30h] [rbp-680h]
  __int64 v66; // [rsp+30h] [rbp-680h]
  __int64 v67; // [rsp+30h] [rbp-680h]
  __int64 v68; // [rsp+38h] [rbp-678h]
  int v69; // [rsp+40h] [rbp-670h]
  int v70; // [rsp+44h] [rbp-66Ch]
  unsigned int v71; // [rsp+44h] [rbp-66Ch]
  unsigned int v72; // [rsp+44h] [rbp-66Ch]
  int v73; // [rsp+48h] [rbp-668h]
  _QWORD v74[2]; // [rsp+50h] [rbp-660h] BYREF
  int v75; // [rsp+60h] [rbp-650h]
  unsigned int v76; // [rsp+64h] [rbp-64Ch]
  void *base; // [rsp+70h] [rbp-640h] BYREF
  __int64 v78; // [rsp+78h] [rbp-638h]
  _BYTE v79[1584]; // [rsp+80h] [rbp-630h] BYREF

  v2 = *(_QWORD *)(a2 + 48);
  v68 = v2;
  v3 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a2 + 16) + 200LL))(*(_QWORD *)(a2 + 16));
  v4 = *(__int64 (**)())(**(_QWORD **)(a2 + 16) + 136LL);
  if ( v4 == sub_2DD19D0 )
    BUG();
  v5 = v4();
  v6 = *(_QWORD *)(a2 + 328);
  v7 = a2 + 320;
  v69 = *(_DWORD *)(v5 + 8);
  base = v79;
  v78 = 0x4000000000LL;
  if ( v6 == a2 + 320 )
  {
    LODWORD(v13) = 0;
    return (unsigned int)v13;
  }
  v8 = 0;
  v9 = v6;
  v10 = v2;
  v11 = 0x11407C000LL;
  do
  {
    v12 = *(_QWORD *)(v9 + 56);
    v13 = v9 + 48;
    if ( v12 != v9 + 48 )
    {
      p_base = &base;
      do
      {
        v15 = *(unsigned __int16 *)(v12 + 68);
        if ( (unsigned __int16)v15 > 0x20u || !_bittest64(&v11, v15) )
        {
          v16 = *(_QWORD *)(v12 + 32);
          v17 = v16 + 40LL * (*(_DWORD *)(v12 + 40) & 0xFFFFFF);
          if ( v17 != v16 )
          {
            while ( *(_BYTE *)v16 != 5 )
            {
              v16 += 40;
              if ( v17 == v16 )
                goto LABEL_13;
            }
            v18 = *(_DWORD *)(v16 + 24);
            if ( *(_BYTE *)(*(_QWORD *)(v10 + 8) + 40LL * (unsigned int)(*(_DWORD *)(v10 + 32) + v18) + 32) )
            {
              v19 = *(__int64 (**)())(*(_QWORD *)v3 + 560LL);
              if ( v19 != sub_2E2CB50 )
              {
                v52 = (__int64)p_base;
                v54 = v7;
                v58 = v10;
                v71 = v8;
                v66 = *(_QWORD *)(*a1 + 8LL * v18);
                v43 = ((__int64 (__fastcall *)(__int64, __int64, __int64, __int64, __int64, __int64))v19)(
                        v3,
                        v12,
                        v66,
                        v8,
                        v10,
                        0x11407C000LL);
                v8 = v71;
                v10 = v58;
                v11 = 0x11407C000LL;
                v7 = v54;
                p_base = (void **)v52;
                if ( v43 )
                {
                  v76 = v71;
                  v72 = v71 + 1;
                  v44 = (unsigned int)v78;
                  v74[1] = v66;
                  v75 = v18;
                  v45 = (char *)base;
                  v74[0] = v12;
                  v46 = (const __m128i *)v74;
                  if ( (unsigned __int64)(unsigned int)v78 + 1 > HIDWORD(v78) )
                  {
                    if ( base > v74 || v74 >= (_QWORD *)((char *)base + 24 * (unsigned int)v78) )
                    {
                      sub_C8D5F0(v52, v79, (unsigned int)v78 + 1LL, 0x18u, v58, 0x11407C000LL);
                      v45 = (char *)base;
                      v44 = (unsigned int)v78;
                      v11 = 0x11407C000LL;
                      v46 = (const __m128i *)v74;
                      v7 = v54;
                      v10 = v58;
                      p_base = (void **)v52;
                    }
                    else
                    {
                      v50 = (char *)v74 - (_BYTE *)base;
                      sub_C8D5F0(v52, v79, (unsigned int)v78 + 1LL, 0x18u, v58, 0x11407C000LL);
                      v45 = (char *)base;
                      v44 = (unsigned int)v78;
                      v11 = 0x11407C000LL;
                      p_base = (void **)v52;
                      v10 = v58;
                      v7 = v54;
                      v46 = (const __m128i *)((char *)base + v50);
                    }
                  }
                  v8 = v72;
                  v47 = (__m128i *)&v45[24 * v44];
                  *v47 = _mm_loadu_si128(v46);
                  v48 = v46[1].m128i_i64[0];
                  LODWORD(v78) = v78 + 1;
                  v47[1].m128i_i64[0] = v48;
                }
              }
            }
          }
        }
LABEL_13:
        if ( (*(_BYTE *)v12 & 4) == 0 )
        {
          while ( (*(_BYTE *)(v12 + 44) & 8) != 0 )
            v12 = *(_QWORD *)(v12 + 8);
        }
        v12 = *(_QWORD *)(v12 + 8);
      }
      while ( v13 != v12 );
    }
    v9 = *(_QWORD *)(v9 + 8);
  }
  while ( v7 != v9 );
  v20 = base;
  if ( 3 * (unsigned __int64)(unsigned int)v78 <= 3 )
  {
    v70 = v78;
    v51 = *(_QWORD *)(a2 + 328);
  }
  else
  {
    qsort(base, 0xAAAAAAAAAAAAAAABLL * ((24LL * (unsigned int)v78) >> 3), 0x18u, sub_2E2D450);
    v20 = base;
    v51 = *(_QWORD *)(a2 + 328);
    v70 = v78;
  }
  if ( v70 <= 0 )
  {
    LODWORD(v13) = 0;
    goto LABEL_48;
  }
  v73 = 0;
  v21 = 0;
  v22 = 0;
  v23 = v20;
  v62 = 0;
  while ( 1 )
  {
    v24 = &v23[v21];
    ++v73;
    v25 = *(_DWORD *)&v23[v21 + 16];
    v26 = *(_DWORD *)(v68 + 68);
    if ( v26 == v25 && v26 != -1 )
      goto LABEL_38;
    v27 = *v24;
    LODWORD(v13) = *(_DWORD *)(*v24 + 40LL) & 0xFFFFFF;
    if ( (_DWORD)v13 )
    {
      v28 = *(_QWORD *)(v27 + 32);
      v29 = 0;
      while ( *(_BYTE *)v28 != 5 || v25 != *(_DWORD *)(v28 + 24) )
      {
        ++v29;
        v28 += 40;
        if ( (_DWORD)v13 == v29 )
          goto LABEL_27;
      }
      LODWORD(v13) = v29;
    }
LABEL_27:
    v30 = v24[1];
    v31 = 0;
    v32 = v30;
    if ( v69 == 1 )
    {
      v31 = *(_QWORD *)(v68 + 656);
      v32 = v30 + v31;
    }
    if ( !v22 )
      break;
    v33 = v32 - v62;
    v60 = v22;
    v63 = *v24;
    v53 = v24[1];
    v56 = v31;
    v34 = (*(__int64 (__fastcall **)(__int64, _QWORD, _QWORD, __int64))(*(_QWORD *)v3 + 584LL))(v3, *v24, v22, v33);
    v27 = v63;
    v22 = v60;
    if ( !v34 )
    {
      v31 = v56;
      v30 = v53;
      break;
    }
LABEL_37:
    v65 = v22;
    (*(void (__fastcall **)(__int64, __int64, _QWORD, __int64))(*(_QWORD *)v3 + 576LL))(v3, v27, v22, v33);
    v23 = base;
    v22 = v65;
LABEL_38:
    if ( v70 <= v73 )
      goto LABEL_47;
    v39 = v21 + 24;
LABEL_40:
    v21 = v39;
  }
  v35 = 0;
  v36 = *(__int64 (**)())(*(_QWORD *)v3 + 552LL);
  if ( v36 != sub_2E2CB40 )
  {
    v55 = v22;
    v59 = v30;
    v61 = v31;
    v67 = v27;
    v49 = ((__int64 (__fastcall *)(__int64, __int64, _QWORD))v36)(v3, v27, (unsigned int)v13);
    v22 = v55;
    v30 = v59;
    v31 = v61;
    v27 = v67;
    v35 = v49;
  }
  v23 = base;
  if ( v70 > v73 )
  {
    v64 = v22;
    v13 = v31 + v30 + v35;
    v57 = v27;
    v37 = (*(__int64 (__fastcall **)(__int64, _QWORD, _QWORD, __int64))(*(_QWORD *)v3 + 584LL))(
            v3,
            *(_QWORD *)((char *)base + v21 + 24),
            v22,
            *(_QWORD *)((char *)base + v21 + 32) - (v30 + v35));
    v22 = v64;
    v39 = v21 + 24;
    if ( !v37 )
    {
      v23 = base;
      goto LABEL_40;
    }
    v40 = v35;
    v33 = -v35;
    v41 = (*(__int64 (__fastcall **)(__int64, __int64, _QWORD, __int64, __int64, _QWORD))(*(_QWORD *)v3 + 568LL))(
            v3,
            v51,
            v25,
            v40,
            v38,
            v64);
    v62 = v13;
    v27 = v57;
    v22 = v41;
    goto LABEL_37;
  }
LABEL_47:
  v20 = v23;
  LOBYTE(v13) = v22 != 0;
LABEL_48:
  if ( v20 != v79 )
    _libc_free((unsigned __int64)v20);
  return (unsigned int)v13;
}
