// Function: sub_1DD32B0
// Address: 0x1dd32b0
//
__int64 __fastcall sub_1DD32B0(__int64 a1, _QWORD *a2)
{
  __int64 v2; // rbx
  __int64 *v3; // rdi
  __int64 v4; // rax
  __int64 (*v5)(void); // rdx
  __int64 (*v6)(); // rax
  __int64 v7; // rcx
  _QWORD *v8; // r15
  __int64 v9; // r8
  void *v10; // r10
  unsigned int v11; // ebx
  __int64 v12; // r12
  unsigned __int64 v13; // rax
  int v14; // edx
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 (*v17)(); // rax
  _BYTE *v18; // rdi
  void *v19; // rbx
  __int64 v20; // r13
  __int64 *v21; // r12
  unsigned __int64 v22; // rax
  __int64 *v23; // r13
  __int64 *v24; // rdi
  __int64 v25; // r14
  __int64 v26; // r15
  __int64 v27; // r9
  __int64 v28; // r11
  unsigned int v29; // r13d
  unsigned int v30; // r12d
  __int64 v31; // rax
  unsigned int v32; // edx
  __int64 v33; // r10
  __int64 v34; // rcx
  unsigned __int8 v35; // al
  __int64 v36; // r8
  __int64 (*v37)(); // rax
  unsigned __int8 v38; // r12
  __int64 v39; // rax
  __int64 v40; // rax
  void *v41; // rdx
  __int64 v42; // rax
  char v44; // al
  int v45; // r9d
  __int64 v46; // rax
  __m128i *v47; // rax
  unsigned __int64 v48; // rdx
  __int64 v49; // [rsp+10h] [rbp-6C0h]
  __int64 v50; // [rsp+18h] [rbp-6B8h]
  __int64 v51; // [rsp+20h] [rbp-6B0h]
  __int64 v52; // [rsp+28h] [rbp-6A8h]
  __int64 v53; // [rsp+28h] [rbp-6A8h]
  __int64 v54; // [rsp+28h] [rbp-6A8h]
  __int64 v55; // [rsp+38h] [rbp-698h]
  __int64 v56; // [rsp+38h] [rbp-698h]
  __int64 v57; // [rsp+40h] [rbp-690h]
  __int64 v58; // [rsp+40h] [rbp-690h]
  __int64 v59; // [rsp+40h] [rbp-690h]
  unsigned int v60; // [rsp+40h] [rbp-690h]
  int v61; // [rsp+4Ch] [rbp-684h]
  __int64 v62; // [rsp+50h] [rbp-680h]
  __int64 v63; // [rsp+50h] [rbp-680h]
  __int64 v64; // [rsp+50h] [rbp-680h]
  __int64 v65; // [rsp+50h] [rbp-680h]
  int v66; // [rsp+58h] [rbp-678h]
  unsigned __int8 v68; // [rsp+60h] [rbp-670h]
  __int64 *src; // [rsp+68h] [rbp-668h]
  int srca; // [rsp+68h] [rbp-668h]
  void *srcb; // [rsp+68h] [rbp-668h]
  __m128i v72; // [rsp+70h] [rbp-660h] BYREF
  unsigned __int64 v73; // [rsp+80h] [rbp-650h]
  _BYTE *v74; // [rsp+90h] [rbp-640h] BYREF
  __int64 v75; // [rsp+98h] [rbp-638h]
  _BYTE v76[1584]; // [rsp+A0h] [rbp-630h] BYREF

  v2 = 0;
  v3 = (__int64 *)a2[2];
  v4 = *v3;
  v49 = a2[7];
  v5 = *(__int64 (**)(void))(*v3 + 112);
  if ( v5 != sub_1D00B10 )
  {
    v2 = v5();
    v4 = *(_QWORD *)a2[2];
  }
  v6 = *(__int64 (**)())(v4 + 48);
  if ( v6 == sub_1D90020 )
    BUG();
  v61 = *(_DWORD *)(v6() + 8);
  v74 = v76;
  v75 = 0x4000000000LL;
  v8 = (_QWORD *)a2[41];
  if ( v8 == a2 + 40 )
  {
    return 0;
  }
  else
  {
    v9 = (__int64)(a2 + 40);
    v10 = (void *)v2;
    v11 = 0;
    do
    {
      v12 = v8[4];
      if ( (_QWORD *)v12 != v8 + 3 )
      {
        do
        {
          v13 = **(unsigned __int16 **)(v12 + 16);
          if ( (unsigned __int16)v13 > 0x17u || (v41 = &loc_A83000, !_bittest64((const __int64 *)&v41, v13)) )
          {
            v14 = *(_DWORD *)(v12 + 40);
            if ( v14 )
            {
              v15 = *(_QWORD *)(v12 + 32);
              v16 = v15 + 40LL * (unsigned int)(v14 - 1) + 40;
              while ( *(_BYTE *)v15 != 5 )
              {
                v15 += 40;
                if ( v16 == v15 )
                  goto LABEL_14;
              }
              v7 = *(unsigned int *)(v15 + 24);
              if ( *(_BYTE *)(*(_QWORD *)(v49 + 8)
                            + 40LL * (unsigned int)(*(_DWORD *)(v49 + 32) + *(_DWORD *)(v15 + 24))
                            + 32) )
              {
                v17 = *(__int64 (**)())(*(_QWORD *)v10 + 352LL);
                if ( v17 != sub_1DD2780 )
                {
                  v56 = v9;
                  srcb = v10;
                  v60 = v7;
                  v65 = *(_QWORD *)(*(_QWORD *)(a1 + 232) + 8LL * (int)v7);
                  v44 = ((__int64 (__fastcall *)(void *, __int64))v17)(v10, v12);
                  v10 = srcb;
                  v9 = v56;
                  if ( v44 )
                  {
                    v7 = v60;
                    v72.m128i_i64[0] = v12;
                    v46 = (unsigned int)v75;
                    v72.m128i_i64[1] = v65;
                    v73 = __PAIR64__(v11, v60);
                    if ( (unsigned int)v75 >= HIDWORD(v75) )
                    {
                      sub_16CD150((__int64)&v74, v76, 0, 24, v56, v45);
                      v46 = (unsigned int)v75;
                      v9 = v56;
                      v10 = srcb;
                    }
                    ++v11;
                    v47 = (__m128i *)&v74[24 * v46];
                    v48 = v73;
                    *v47 = _mm_loadu_si128(&v72);
                    v47[1].m128i_i64[0] = v48;
                    LODWORD(v75) = v75 + 1;
                  }
                }
              }
            }
          }
LABEL_14:
          if ( (*(_BYTE *)v12 & 4) == 0 )
          {
            while ( (*(_BYTE *)(v12 + 46) & 8) != 0 )
              v12 = *(_QWORD *)(v12 + 8);
          }
          v12 = *(_QWORD *)(v12 + 8);
        }
        while ( v8 + 3 != (_QWORD *)v12 );
      }
      v8 = (_QWORD *)v8[1];
    }
    while ( (_QWORD *)v9 != v8 );
    v18 = v74;
    v19 = v10;
    v20 = 24LL * (unsigned int)v75;
    v21 = (__int64 *)&v74[v20];
    if ( &v74[v20] == v74 )
    {
      v66 = v75;
      v50 = a2[41];
    }
    else
    {
      src = (__int64 *)v74;
      _BitScanReverse64(&v22, 0xAAAAAAAAAAAAAAABLL * (v20 >> 3));
      sub_1DD2ED0((__int64)v74, (__m128i *)&v74[v20], 2LL * (int)(63 - (v22 ^ 0x3F)), v7, v9);
      if ( (unsigned __int64)v20 <= 0x180 )
      {
        sub_1DD2A20(src, v21);
      }
      else
      {
        v23 = src + 48;
        sub_1DD2A20(src, src + 48);
        if ( v21 != src + 48 )
        {
          do
          {
            v24 = v23;
            v23 += 3;
            sub_1DD29D0(v24);
          }
          while ( v21 != v23 );
        }
      }
      v18 = v74;
      v50 = a2[41];
      v66 = v75;
    }
    if ( v66 > 0 )
    {
      srca = 0;
      v25 = 0;
      LODWORD(v26) = 0;
      v55 = 0;
      v68 = 0;
      while ( 1 )
      {
        v27 = *(_QWORD *)&v18[v25];
        v28 = *(_QWORD *)&v18[v25 + 8];
        v29 = *(_DWORD *)&v18[v25 + 16];
        v30 = *(_DWORD *)(v27 + 40);
        if ( v30 )
        {
          v31 = *(_QWORD *)(v27 + 32);
          v32 = 0;
          while ( *(_BYTE *)v31 != 5 || v29 != *(_DWORD *)(v31 + 24) )
          {
            ++v32;
            v31 += 40;
            if ( v32 == v30 )
              goto LABEL_29;
          }
          v30 = v32;
        }
LABEL_29:
        v33 = 0;
        v34 = *(_QWORD *)&v18[v25 + 8];
        if ( v61 == 1 )
        {
          v33 = *(_QWORD *)(v49 + 640);
          v34 = v28 + v33;
        }
        ++srca;
        if ( !v68 )
          goto LABEL_34;
        v57 = v34 - v55;
        v62 = *(_QWORD *)&v18[v25];
        v51 = *(_QWORD *)&v18[v25 + 8];
        v52 = v33;
        v35 = (*(__int64 (__fastcall **)(void *, __int64, _QWORD))(*(_QWORD *)v19 + 376LL))(v19, v27, (unsigned int)v26);
        v27 = v62;
        v34 = v57;
        if ( !v35 )
          break;
        v68 = v35;
        (*(void (__fastcall **)(void *, __int64, _QWORD, __int64))(*(_QWORD *)v19 + 368LL))(
          v19,
          v62,
          (unsigned int)v26,
          v57);
        if ( v66 <= srca )
        {
          v18 = v74;
          goto LABEL_49;
        }
        v25 += 24;
LABEL_39:
        v18 = v74;
      }
      v33 = v52;
      v28 = v51;
LABEL_34:
      v36 = 0;
      v37 = *(__int64 (**)())(*(_QWORD *)v19 + 344LL);
      if ( v37 != sub_1DD2770 )
      {
        v54 = v28;
        v59 = v33;
        v64 = v27;
        v42 = ((__int64 (__fastcall *)(void *, __int64, _QWORD, __int64, _QWORD))v37)(v19, v27, v30, v34, 0);
        v28 = v54;
        v33 = v59;
        v27 = v64;
        v36 = v42;
      }
      v58 = v27;
      v18 = v74;
      if ( v66 <= srca )
        goto LABEL_49;
      v25 += 24;
      v53 = v36;
      v63 = v33 + v28 + v36;
      v38 = (*(__int64 (__fastcall **)(void *, _QWORD, _QWORD, __int64))(*(_QWORD *)v19 + 376LL))(
              v19,
              *(_QWORD *)&v74[v25],
              (unsigned int)v26,
              *(_QWORD *)&v74[v25 + 8] - (v28 + v36));
      if ( v38 )
      {
        v39 = sub_1E15F70(v58);
        v40 = (*(__int64 (__fastcall **)(void *, __int64, _QWORD))(*(_QWORD *)v19 + 144LL))(v19, v39, 0);
        v26 = (unsigned int)sub_1E6B9A0(a2[5], v40, byte_3F871B3, 0);
        (*(void (__fastcall **)(void *, __int64, __int64, _QWORD))(*(_QWORD *)v19 + 360LL))(v19, v50, v26, v29);
        (*(void (__fastcall **)(void *, __int64, _QWORD, __int64))(*(_QWORD *)v19 + 368LL))(
          v19,
          v58,
          (unsigned int)v26,
          -v53);
        v68 = v38;
        v55 = v63;
      }
      goto LABEL_39;
    }
    v68 = 0;
LABEL_49:
    if ( v18 != v76 )
      _libc_free((unsigned __int64)v18);
  }
  return v68;
}
