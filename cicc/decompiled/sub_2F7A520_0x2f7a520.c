// Function: sub_2F7A520
// Address: 0x2f7a520
//
void __fastcall sub_2F7A520(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 *v6; // rax
  __int64 v7; // rbx
  __int64 *v8; // r13
  __int64 *v9; // r12
  __int64 **v10; // rdx
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 **v13; // r13
  __int64 v14; // r12
  unsigned __int64 v15; // rax
  __int64 **v16; // rbx
  __int64 *v17; // r13
  const char *v18; // rax
  size_t v19; // rdx
  size_t v20; // r12
  const char *v21; // r14
  const char *v22; // rax
  size_t v23; // rdx
  size_t v24; // r15
  int v25; // eax
  __int64 *v26; // rax
  unsigned __int64 v27; // rdx
  __int64 *v28; // r12
  const char *v29; // rax
  size_t v30; // rdx
  _BYTE *v31; // rdi
  unsigned __int8 *v32; // rsi
  _BYTE *v33; // rax
  size_t v34; // r15
  __int64 v35; // r14
  __int64 v36; // rdx
  __m128i si128; // xmm0
  __int64 v38; // rdi
  __int64 (*v39)(); // rax
  unsigned int v40; // r15d
  __int64 v41; // rax
  __int64 v42; // rax
  unsigned int v43; // r14d
  int v44; // eax
  __int64 v45; // rdx
  _BYTE *v46; // rax
  _BYTE *v47; // rax
  __int64 v48; // rax
  __int64 **v50; // [rsp+18h] [rbp-278h]
  __int64 v51; // [rsp+20h] [rbp-270h]
  __int64 **v52; // [rsp+28h] [rbp-268h]
  __int64 **v53; // [rsp+28h] [rbp-268h]
  _BYTE v54[16]; // [rsp+30h] [rbp-260h] BYREF
  void (__fastcall *v55)(_BYTE *, _BYTE *, __int64); // [rsp+40h] [rbp-250h]
  void (__fastcall *v56)(_BYTE *, __int64); // [rsp+48h] [rbp-248h]
  __int64 **v57; // [rsp+50h] [rbp-240h] BYREF
  __int64 v58; // [rsp+58h] [rbp-238h]
  _BYTE v59[560]; // [rsp+60h] [rbp-230h] BYREF

  v57 = (__int64 **)v59;
  v58 = 0x4000000000LL;
  if ( !*(_DWORD *)(a1 + 16) )
    goto LABEL_2;
  v6 = *(__int64 **)(a1 + 8);
  v7 = a2;
  v8 = &v6[4 * *(unsigned int *)(a1 + 24)];
  if ( v6 == v8 )
    goto LABEL_2;
  while ( 1 )
  {
    v9 = v6;
    if ( *v6 != -4096 && *v6 != -8192 )
      break;
    v6 += 4;
    if ( v8 == v6 )
      goto LABEL_2;
  }
  if ( v8 == v6 )
  {
LABEL_2:
    v50 = (__int64 **)v59;
  }
  else
  {
    v10 = (__int64 **)v59;
    v11 = 0;
    while ( 1 )
    {
      v10[v11] = v9;
      v9 += 4;
      v12 = (unsigned int)(v58 + 1);
      LODWORD(v58) = v58 + 1;
      if ( v9 == v8 )
        break;
      while ( *v9 == -8192 || *v9 == -4096 )
      {
        v9 += 4;
        if ( v8 == v9 )
          goto LABEL_17;
      }
      v11 = (unsigned int)v12;
      v12 = (unsigned int)v12;
      if ( v8 == v9 )
        break;
      v27 = (unsigned int)v12 + 1LL;
      if ( v11 + 1 > (unsigned __int64)HIDWORD(v58) )
      {
        sub_C8D5F0((__int64)&v57, v59, v27, 8u, a5, a6);
        v11 = (unsigned int)v58;
      }
      v10 = v57;
    }
LABEL_17:
    v13 = v57;
    v14 = 8 * v12;
    v50 = &v57[v12];
    if ( v57 != v50 )
    {
      _BitScanReverse64(&v15, v14 >> 3);
      sub_2F79EB0(v57, v50, 2LL * (int)(63 - (v15 ^ 0x3F)));
      if ( (unsigned __int64)v14 <= 0x80 )
      {
        sub_2F79B50(v13, v50);
        goto LABEL_37;
      }
      v52 = v13 + 16;
      sub_2F79B50(v13, v13 + 16);
      if ( v50 != v13 + 16 )
      {
LABEL_22:
        v16 = v52;
        v17 = *v52;
        while ( 1 )
        {
          v18 = sub_BD5D20(**(v16 - 1));
          v20 = v19;
          v21 = v18;
          v22 = sub_BD5D20(*v17);
          v24 = v23;
          if ( v20 <= v23 )
            v23 = v20;
          if ( v23 && (v25 = memcmp(v22, v21, v23)) != 0 )
          {
            if ( v25 >= 0 )
              goto LABEL_21;
          }
          else if ( v20 == v24 || v20 <= v24 )
          {
LABEL_21:
            ++v52;
            *v16 = v17;
            if ( v50 != v52 )
              goto LABEL_22;
            v7 = a2;
            break;
          }
          v26 = *--v16;
          v16[1] = v26;
        }
      }
LABEL_37:
      v50 = &v57[(unsigned int)v58];
      if ( v50 != v57 )
      {
        v53 = v57;
        do
        {
          v28 = *v53;
          v29 = sub_BD5D20(**v53);
          v31 = *(_BYTE **)(v7 + 32);
          v32 = (unsigned __int8 *)v29;
          v33 = *(_BYTE **)(v7 + 24);
          v34 = v30;
          if ( v33 - v31 < v30 )
          {
            v35 = sub_CB6200(v7, v32, v30);
            v33 = *(_BYTE **)(v35 + 24);
            v31 = *(_BYTE **)(v35 + 32);
          }
          else
          {
            v35 = v7;
            if ( v30 )
            {
              memcpy(v31, v32, v30);
              v33 = *(_BYTE **)(v7 + 24);
              v31 = (_BYTE *)(v34 + *(_QWORD *)(v7 + 32));
              *(_QWORD *)(v7 + 32) = v31;
            }
          }
          if ( v33 == v31 )
          {
            v48 = sub_CB6200(v35, (unsigned __int8 *)" ", 1u);
            v36 = *(_QWORD *)(v48 + 32);
            v35 = v48;
          }
          else
          {
            *v31 = 32;
            v36 = *(_QWORD *)(v35 + 32) + 1LL;
            *(_QWORD *)(v35 + 32) = v36;
          }
          if ( (unsigned __int64)(*(_QWORD *)(v35 + 24) - v36) <= 0x14 )
          {
            sub_CB6200(v35, "Clobbered Registers: ", 0x15u);
          }
          else
          {
            si128 = _mm_load_si128((const __m128i *)&xmmword_430B360);
            *(_DWORD *)(v36 + 16) = 980644453;
            *(_BYTE *)(v36 + 20) = 32;
            *(__m128i *)v36 = si128;
            *(_QWORD *)(v35 + 32) += 21LL;
          }
          v38 = *(_QWORD *)(a1 + 32);
          v39 = *(__int64 (**)())(*(_QWORD *)v38 + 16LL);
          if ( v39 == sub_23CE270 )
            BUG();
          v40 = 1;
          v41 = ((__int64 (__fastcall *)(__int64, __int64))v39)(v38, *v28);
          v42 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v41 + 200LL))(v41);
          v43 = *(_DWORD *)(v42 + 16);
          v51 = v42;
          if ( v43 > 1 )
          {
            do
            {
              while ( 1 )
              {
                v44 = *(_DWORD *)(v28[1] + 4LL * (v40 >> 5));
                if ( !_bittest(&v44, v40) )
                {
                  sub_2FF6320(v54, v40, v51, 0, 0);
                  if ( !v55 )
                    sub_4263D6(v54, v40, v45);
                  v56(v54, v7);
                  v46 = *(_BYTE **)(v7 + 32);
                  if ( *(_BYTE **)(v7 + 24) == v46 )
                  {
                    sub_CB6200(v7, (unsigned __int8 *)" ", 1u);
                  }
                  else
                  {
                    *v46 = 32;
                    ++*(_QWORD *)(v7 + 32);
                  }
                  if ( v55 )
                    break;
                }
                if ( ++v40 == v43 )
                  goto LABEL_56;
              }
              ++v40;
              v55(v54, v54, 3);
            }
            while ( v40 != v43 );
          }
LABEL_56:
          v47 = *(_BYTE **)(v7 + 32);
          if ( *(_BYTE **)(v7 + 24) == v47 )
          {
            sub_CB6200(v7, (unsigned __int8 *)"\n", 1u);
          }
          else
          {
            *v47 = 10;
            ++*(_QWORD *)(v7 + 32);
          }
          ++v53;
        }
        while ( v50 != v53 );
        v50 = v57;
      }
    }
  }
  if ( v50 != (__int64 **)v59 )
    _libc_free((unsigned __int64)v50);
}
