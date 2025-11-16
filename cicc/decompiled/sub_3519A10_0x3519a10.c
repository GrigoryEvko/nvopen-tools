// Function: sub_3519A10
// Address: 0x3519a10
//
void __fastcall sub_3519A10(__int64 a1, __int64 **a2)
{
  _BYTE *v2; // r13
  __int64 v4; // r8
  __int64 v5; // r9
  __int64 *v6; // rsi
  __int64 v7; // rcx
  _QWORD *v8; // r14
  __int64 v9; // rax
  __int64 v10; // rdi
  __int64 v11; // rbx
  __int64 v12; // rbx
  char *v13; // rdx
  char *v14; // rax
  char *i; // rdx
  __int64 v16; // rbx
  _QWORD *v17; // r14
  __int64 v18; // rax
  __int64 *v19; // rbx
  __int64 *v20; // r11
  unsigned int v21; // esi
  __int64 v22; // rdi
  unsigned int v23; // edx
  __int64 *v24; // r14
  int v25; // r11d
  _QWORD *v26; // rcx
  unsigned int v27; // r9d
  _QWORD *v28; // rax
  __int64 v29; // r8
  __int64 v30; // r12
  unsigned __int64 v31; // xmm0_8
  __int64 v32; // rax
  unsigned int v33; // edx
  __int64 v34; // rdi
  int v35; // r10d
  _QWORD *v36; // r9
  __int64 v37; // r15
  __int64 v38; // r12
  __int64 (*v39)(void); // rax
  __int64 v40; // rbx
  __int64 v41; // r12
  __int64 v42; // rdx
  __int64 v43; // r14
  __int64 (*v44)(); // rax
  _QWORD *v45; // r8
  __int64 v46; // r13
  int v47; // r9d
  __int64 v48; // rsi
  __int64 v49; // rax
  unsigned __int64 v50; // [rsp+8h] [rbp-158h]
  _BYTE *v51; // [rsp+10h] [rbp-150h]
  char *v52; // [rsp+18h] [rbp-148h]
  __int64 v53; // [rsp+20h] [rbp-140h] BYREF
  __int64 v54; // [rsp+28h] [rbp-138h] BYREF
  __int64 v55; // [rsp+30h] [rbp-130h] BYREF
  __int64 v56; // [rsp+38h] [rbp-128h]
  __int64 v57; // [rsp+40h] [rbp-120h]
  unsigned int v58; // [rsp+48h] [rbp-118h]
  char *v59; // [rsp+50h] [rbp-110h] BYREF
  __int64 v60; // [rsp+58h] [rbp-108h]
  _BYTE v61[32]; // [rsp+60h] [rbp-100h] BYREF
  _BYTE *v62; // [rsp+80h] [rbp-E0h] BYREF
  __int64 v63; // [rsp+88h] [rbp-D8h]
  _BYTE v64[208]; // [rsp+90h] [rbp-D0h] BYREF

  v2 = (_BYTE *)a1;
  sub_2E7A760(*(_QWORD *)(a1 + 520), 0);
  *(_QWORD *)(a1 + 576) = 0;
  v6 = *a2;
  v7 = a2[1] - *a2;
  if ( v7 )
  {
    v8 = *(_QWORD **)(a1 + 520);
    v9 = 0;
    v10 = v8[12];
    while ( v6[v9] == *(_QWORD *)(v10 + 8LL * (unsigned int)v9) )
    {
      if ( ++v9 == v7 )
        return;
    }
    v11 = v8[13];
    v52 = v61;
    v59 = v61;
    v12 = (v11 - v10) >> 3;
    v60 = 0x400000000LL;
    if ( (_DWORD)v12 )
    {
      v13 = v52;
      v14 = v52;
      if ( (unsigned int)v12 > 4uLL )
      {
        sub_C8D5F0((__int64)&v59, v52, (unsigned int)v12, 8u, v4, v5);
        v13 = v59;
        v14 = &v59[8 * (unsigned int)v60];
      }
      for ( i = &v13[8 * (unsigned int)v12]; i != v14; v14 += 8 )
      {
        if ( v14 )
          *(_QWORD *)v14 = 0;
      }
      LODWORD(v60) = v12;
      v8 = (_QWORD *)*((_QWORD *)v2 + 65);
    }
    v16 = v8[41];
    v17 = v8 + 40;
    if ( (_QWORD *)v16 != v17 )
    {
      do
      {
        v18 = sub_2E32300((__int64 *)v16, 1);
        *(_QWORD *)&v59[8 * *(int *)(v16 + 24)] = v18;
        v16 = *(_QWORD *)(v16 + 8);
      }
      while ( v17 != (_QWORD *)v16 );
    }
    v19 = *a2;
    v20 = a2[1];
    v55 = 0;
    v56 = 0;
    v57 = 0;
    v58 = 0;
    if ( v20 != v19 )
    {
      v51 = v2;
      v21 = 0;
      v22 = 0;
      v23 = 0;
      v24 = v20;
      while ( 1 )
      {
        v30 = *v19;
        v31 = _mm_cvtsi32_si128(v23).m128i_u64[0];
        if ( !v21 )
          break;
        v25 = 1;
        v26 = 0;
        v27 = (v21 - 1) & (((unsigned int)v30 >> 9) ^ ((unsigned int)v30 >> 4));
        v28 = (_QWORD *)(v22 + 16LL * v27);
        v29 = *v28;
        if ( v30 == *v28 )
        {
LABEL_18:
          ++v19;
          v28[1] = v31;
          if ( v24 == v19 )
            goto LABEL_47;
          goto LABEL_19;
        }
        while ( v29 != -4096 )
        {
          if ( v29 == -8192 && !v26 )
            v26 = v28;
          v27 = (v21 - 1) & (v25 + v27);
          LODWORD(v50) = v25 + 1;
          v28 = (_QWORD *)(v22 + 16LL * v27);
          v29 = *v28;
          if ( v30 == *v28 )
            goto LABEL_18;
          v25 = v50;
        }
        if ( !v26 )
          v26 = v28;
        v33 = v23 + 1;
        ++v55;
        if ( 4 * v33 >= 3 * v21 )
          goto LABEL_22;
        if ( v21 - HIDWORD(v57) - v33 <= v21 >> 3 )
        {
          v50 = v31;
          sub_2E3E470((__int64)&v55, v21);
          if ( !v58 )
          {
LABEL_77:
            LODWORD(v57) = v57 + 1;
            BUG();
          }
          v45 = 0;
          LODWORD(v46) = (v58 - 1) & (((unsigned int)v30 >> 9) ^ ((unsigned int)v30 >> 4));
          v47 = 1;
          v31 = v50;
          v33 = v57 + 1;
          v26 = (_QWORD *)(v56 + 16LL * (unsigned int)v46);
          v48 = *v26;
          if ( v30 != *v26 )
          {
            while ( v48 != -4096 )
            {
              if ( v48 == -8192 && !v45 )
                v45 = v26;
              v46 = (v58 - 1) & ((_DWORD)v46 + v47);
              v26 = (_QWORD *)(v56 + 16 * v46);
              v48 = *v26;
              if ( v30 == *v26 )
                goto LABEL_44;
              ++v47;
            }
            if ( v45 )
              v26 = v45;
          }
        }
LABEL_44:
        LODWORD(v57) = v33;
        if ( *v26 != -4096 )
          --HIDWORD(v57);
        ++v19;
        *v26 = v30;
        v26[1] = 0;
        v26[1] = v31;
        if ( v24 == v19 )
        {
LABEL_47:
          v2 = v51;
          goto LABEL_48;
        }
LABEL_19:
        v23 = v57;
        v22 = v56;
        v21 = v58;
      }
      ++v55;
LABEL_22:
      v50 = v31;
      sub_2E3E470((__int64)&v55, 2 * v21);
      if ( !v58 )
        goto LABEL_77;
      v31 = v50;
      LODWORD(v32) = (v58 - 1) & (((unsigned int)v30 >> 9) ^ ((unsigned int)v30 >> 4));
      v33 = v57 + 1;
      v26 = (_QWORD *)(v56 + 16LL * (unsigned int)v32);
      v34 = *v26;
      if ( v30 != *v26 )
      {
        v35 = 1;
        v36 = 0;
        while ( v34 != -4096 )
        {
          if ( !v36 && v34 == -8192 )
            v36 = v26;
          v32 = (v58 - 1) & ((_DWORD)v32 + v35);
          v26 = (_QWORD *)(v56 + 16 * v32);
          v34 = *v26;
          if ( v30 == *v26 )
            goto LABEL_44;
          ++v35;
        }
        if ( v36 )
          v26 = v36;
      }
      goto LABEL_44;
    }
LABEL_48:
    v37 = 0;
    sub_3518EF0((unsigned __int64 *)(*((_QWORD *)v2 + 65) + 320LL), (__int64)&v55);
    v38 = *((_QWORD *)v2 + 65);
    v39 = *(__int64 (**)(void))(**(_QWORD **)(v38 + 16) + 128LL);
    if ( v39 != sub_2DAC790 )
    {
      v49 = v39();
      v38 = *((_QWORD *)v2 + 65);
      v37 = v49;
    }
    v40 = *(_QWORD *)(v38 + 328);
    v41 = v38 + 320;
    v51 = v64;
    v62 = v64;
    v63 = 0x400000000LL;
    if ( v41 != v40 )
    {
      do
      {
        v42 = *(_QWORD *)(v40 + 8);
        v43 = *(_QWORD *)&v59[8 * *(int *)(v40 + 24)];
        if ( v43 && (v42 == *(_QWORD *)(v40 + 32) + 320LL || v43 != v42) )
        {
          sub_2E32880(&v54, v40);
          (*(void (__fastcall **)(__int64, __int64, __int64, _QWORD, _QWORD, _QWORD, __int64 *, _QWORD))(*(_QWORD *)v37 + 368LL))(
            v37,
            v40,
            v43,
            0,
            0,
            0,
            &v54,
            0);
          if ( v54 )
            sub_B91220((__int64)&v54, v54);
        }
        v54 = 0;
        LODWORD(v63) = 0;
        v53 = 0;
        v44 = *(__int64 (**)())(*(_QWORD *)v37 + 344LL);
        if ( v44 != sub_2DB1AE0
          && !((unsigned __int8 (__fastcall *)(__int64, __int64, __int64 *, __int64 *, _BYTE **, _QWORD))v44)(
                v37,
                v40,
                &v53,
                &v54,
                &v62,
                0) )
        {
          sub_2E32A60(v40, v43);
        }
        v40 = *(_QWORD *)(v40 + 8);
      }
      while ( v41 != v40 );
      if ( v62 != v51 )
        _libc_free((unsigned __int64)v62);
    }
    sub_C7D6A0(v56, 16LL * v58, 8);
    if ( v59 != v52 )
      _libc_free((unsigned __int64)v59);
  }
}
