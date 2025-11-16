// Function: sub_1E4ADF0
// Address: 0x1e4adf0
//
void __fastcall sub_1E4ADF0(__int64 a1)
{
  __int64 v1; // r15
  __int64 v2; // r13
  __int64 (*v3)(); // rax
  __int64 v4; // rdi
  __int64 v5; // r12
  __int64 (*v6)(); // rdx
  unsigned int v7; // ebx
  __int64 v8; // r14
  __int64 v9; // rax
  unsigned int v10; // eax
  __int64 v11; // rax
  __int64 v12; // r14
  __int64 v13; // rdi
  __int64 (*v14)(); // rax
  __int64 v15; // rdi
  __int64 (*v16)(); // rax
  __int64 v17; // rbx
  __int64 v18; // rdx
  __int64 v19; // rcx
  _QWORD *v20; // rsi
  int v21; // r9d
  __int64 v22; // r8
  __int64 v23; // rdi
  __int64 (*v24)(); // rax
  char v25; // bl
  __int64 v26; // rdx
  __int64 v27; // rcx
  int v28; // r8d
  int v29; // r9d
  __int64 v30; // rsi
  __int64 v31; // rbx
  __int64 v32; // rsi
  __int64 v33; // rax
  __int64 v34; // r12
  __int64 v35; // r14
  int v36; // r9d
  __int64 v37; // rcx
  const __m128i *v38; // rax
  const __m128i *v39; // r8
  __m128i v40; // xmm0
  __int64 v41; // r12
  __int64 v42; // rbx
  __int64 v43; // rcx
  const __m128i *v44; // rbx
  const __m128i *v45; // r8
  __m128i v46; // xmm1
  __int64 v47; // r13
  __int64 v48; // rbx
  unsigned int v49; // esi
  __int64 v50; // rdi
  unsigned int v51; // edx
  __int64 *v52; // rax
  __int64 v53; // rcx
  int v54; // r11d
  __int64 *v55; // r9
  int v56; // ecx
  int v57; // ecx
  __int64 v58; // rdi
  const __m128i *v59; // [rsp+0h] [rbp-F0h]
  __int64 v60; // [rsp+18h] [rbp-D8h]
  __int64 v61; // [rsp+20h] [rbp-D0h]
  __int64 v62; // [rsp+20h] [rbp-D0h]
  const __m128i *v63; // [rsp+20h] [rbp-D0h]
  const __m128i *v64; // [rsp+20h] [rbp-D0h]
  unsigned int v65; // [rsp+44h] [rbp-ACh]
  __int64 v66; // [rsp+48h] [rbp-A8h]
  __int64 v67; // [rsp+50h] [rbp-A0h] BYREF
  __int64 *v68; // [rsp+58h] [rbp-98h] BYREF
  unsigned __int64 v69; // [rsp+60h] [rbp-90h] BYREF
  unsigned int v70; // [rsp+68h] [rbp-88h]
  int v71; // [rsp+6Ch] [rbp-84h]
  _BYTE *v72; // [rsp+70h] [rbp-80h] BYREF
  __int64 v73; // [rsp+78h] [rbp-78h]
  _BYTE v74[112]; // [rsp+80h] [rbp-70h] BYREF

  v1 = *(_QWORD *)(a1 + 48);
  v66 = *(_QWORD *)(a1 + 56);
  if ( v66 != v1 )
  {
    v2 = a1;
    while ( 1 )
    {
      v4 = *(_QWORD *)(v2 + 16);
      v5 = *(_QWORD *)(v1 + 8);
      v6 = *(__int64 (**)())(*(_QWORD *)v4 + 648LL);
      if ( v6 == sub_1E40470 )
      {
        v3 = *(__int64 (**)())(*(_QWORD *)v4 + 600LL);
        if ( v3 != sub_1E40450 )
          goto LABEL_8;
LABEL_4:
        v1 += 272;
        if ( v66 == v1 )
          return;
      }
      else
      {
        if ( ((unsigned __int8 (__fastcall *)(__int64, _QWORD))v6)(v4, *(_QWORD *)(v1 + 8)) )
          goto LABEL_4;
        v4 = *(_QWORD *)(v2 + 16);
        v3 = *(__int64 (**)())(*(_QWORD *)v4 + 600LL);
        if ( v3 == sub_1E40450 )
          goto LABEL_4;
LABEL_8:
        if ( !((unsigned __int8 (__fastcall *)(__int64, __int64, __int64 *))v3)(v4, v5, &v67) )
          goto LABEL_4;
        v7 = *(_DWORD *)(*(_QWORD *)(v5 + 32) + 40LL * (unsigned int)v67 + 8);
        v8 = *(_QWORD *)(sub_1E15F70(v5) + 40);
        v9 = sub_1E69D00(v8, v7);
        if ( !v9 || **(_WORD **)(v9 + 16) != 45 && **(_WORD **)(v9 + 16) )
          goto LABEL_4;
        v10 = sub_1E40FE0(*(_QWORD *)(v9 + 32), *(_DWORD *)(v9 + 40), *(_QWORD *)(v5 + 24));
        v65 = v10;
        if ( !v10 )
          goto LABEL_4;
        v11 = sub_1E69D00(v8, v10);
        v12 = v11;
        if ( !v11 )
          goto LABEL_4;
        if ( v5 == v11 )
          goto LABEL_4;
        v13 = *(_QWORD *)(v2 + 16);
        v14 = *(__int64 (**)())(*(_QWORD *)v13 + 648LL);
        if ( v14 == sub_1E40470 )
          goto LABEL_4;
        if ( !((unsigned __int8 (__fastcall *)(__int64, __int64))v14)(v13, v12) )
          goto LABEL_4;
        v15 = *(_QWORD *)(v2 + 16);
        LODWORD(v72) = 0;
        LODWORD(v69) = 0;
        v16 = *(__int64 (**)())(*(_QWORD *)v15 + 600LL);
        if ( v16 == sub_1E40450
          || !((unsigned __int8 (__fastcall *)(__int64, __int64, unsigned __int64 *))v16)(v15, v12, &v69) )
        {
          goto LABEL_4;
        }
        v17 = *(_QWORD *)(*(_QWORD *)(v5 + 32) + 40LL * (unsigned int)v68 + 24);
        v60 = *(_QWORD *)(*(_QWORD *)(v12 + 32) + 40LL * (unsigned int)v72 + 24);
        v20 = sub_1E0B7C0(*(_QWORD *)(v2 + 32), v5);
        v22 = v20[4];
        *(_QWORD *)(v22 + 40LL * (unsigned int)v68 + 24) = v60 + v17;
        v23 = *(_QWORD *)(v2 + 16);
        v24 = *(__int64 (**)())(*(_QWORD *)v23 + 952LL);
        if ( v24 != sub_1E15BA0 )
        {
          v25 = ((__int64 (__fastcall *)(__int64, _QWORD *, __int64, _QWORD))v24)(v23, v20, v12, 0);
          sub_1E0A0F0(*(_QWORD *)(v2 + 32), (__int64)v20, v26, v27, v28, v29);
          if ( !v25 )
            goto LABEL_4;
          v30 = sub_1E69D60(*(_QWORD *)(v2 + 40));
          if ( !v30 )
            goto LABEL_4;
          v31 = sub_1E45EB0(v2, v30);
          if ( !v31 )
            goto LABEL_4;
          v32 = sub_1E69D60(*(_QWORD *)(v2 + 40));
          if ( !v32 )
            goto LABEL_4;
          v33 = sub_1E45EB0(v2, v32);
          v34 = v33;
          if ( !v33 )
            goto LABEL_4;
          v35 = v2 + 2144;
          if ( (unsigned __int8)sub_1F03240(v2 + 2144, v1, v33) )
            goto LABEL_4;
          v37 = 0;
          v72 = v74;
          v73 = 0x400000000LL;
          v38 = *(const __m128i **)(v1 + 32);
          v39 = &v38[*(unsigned int *)(v1 + 40)];
          if ( v38 != v39 )
          {
            do
            {
              while ( v31 != (v38->m128i_i64[0] & 0xFFFFFFFFFFFFFFF8LL) )
              {
                if ( ++v38 == v39 )
                  goto LABEL_35;
              }
              if ( HIDWORD(v73) <= (unsigned int)v37 )
              {
                v59 = v39;
                v63 = v38;
                sub_16CD150((__int64)&v72, v74, 0, 16, (int)v39, v36);
                v37 = (unsigned int)v73;
                v39 = v59;
                v38 = v63;
              }
              v40 = _mm_loadu_si128(v38++);
              *(__m128i *)&v72[16 * v37] = v40;
              v37 = (unsigned int)(v73 + 1);
              LODWORD(v73) = v73 + 1;
            }
            while ( v38 != v39 );
LABEL_35:
            if ( (_DWORD)v37 )
            {
              v61 = v34;
              v41 = 0;
              v42 = 16 * ((unsigned int)(v37 - 1) + 1LL);
              do
              {
                nullsub_752(v2 + 2144, v1, *(_QWORD *)&v72[v41] & 0xFFFFFFFFFFFFFFF8LL);
                v41 += 16;
                sub_1F01C30(v1);
              }
              while ( v42 != v41 );
              v34 = v61;
            }
          }
          LODWORD(v73) = 0;
          v43 = 0;
          v44 = *(const __m128i **)(v34 + 32);
          v45 = &v44[*(unsigned int *)(v34 + 40)];
          if ( v45 != v44 )
          {
            do
            {
              while ( v1 != (v44->m128i_i64[0] & 0xFFFFFFFFFFFFFFF8LL) || ((v44->m128i_i8[0] ^ 6) & 6) != 0 )
              {
                if ( v45 == ++v44 )
                  goto LABEL_47;
              }
              if ( HIDWORD(v73) <= (unsigned int)v43 )
              {
                v64 = v45;
                sub_16CD150((__int64)&v72, v74, 0, 16, (int)v45, v36);
                v43 = (unsigned int)v73;
                v45 = v64;
              }
              v46 = _mm_loadu_si128(v44++);
              *(__m128i *)&v72[16 * v43] = v46;
              v43 = (unsigned int)(v73 + 1);
              LODWORD(v73) = v73 + 1;
            }
            while ( v45 != v44 );
LABEL_47:
            if ( (_DWORD)v43 )
            {
              v62 = v2;
              v47 = 0;
              v48 = 16 * ((unsigned int)(v43 - 1) + 1LL);
              do
              {
                nullsub_752(v35, v34, *(_QWORD *)&v72[v47] & 0xFFFFFFFFFFFFFFF8LL);
                v47 += 16;
                sub_1F01C30(v34);
              }
              while ( v48 != v47 );
              v2 = v62;
            }
          }
          v71 = 0;
          v69 = v1 & 0xFFFFFFFFFFFFFFF9LL | 2;
          v70 = v65;
          sub_1F01A00(v34, &v69, 1);
          v49 = *(_DWORD *)(v2 + 2336);
          v67 = v1;
          if ( v49 )
          {
            v50 = *(_QWORD *)(v2 + 2320);
            v51 = (v49 - 1) & (((unsigned int)v1 >> 9) ^ ((unsigned int)v1 >> 4));
            v52 = (__int64 *)(v50 + 24LL * v51);
            v53 = *v52;
            if ( v1 == *v52 )
            {
LABEL_53:
              *((_DWORD *)v52 + 2) = v65;
              v52[2] = v60;
              if ( v72 != v74 )
                _libc_free((unsigned __int64)v72);
              goto LABEL_4;
            }
            v54 = 1;
            v55 = 0;
            while ( v53 != -8 )
            {
              if ( !v55 && v53 == -16 )
                v55 = v52;
              v51 = (v49 - 1) & (v54 + v51);
              v52 = (__int64 *)(v50 + 24LL * v51);
              v53 = *v52;
              if ( *v52 == v1 )
                goto LABEL_53;
              ++v54;
            }
            v56 = *(_DWORD *)(v2 + 2328);
            if ( v55 )
              v52 = v55;
            ++*(_QWORD *)(v2 + 2312);
            v57 = v56 + 1;
            if ( 4 * v57 < 3 * v49 )
            {
              v58 = v1;
              if ( v49 - *(_DWORD *)(v2 + 2332) - v57 > v49 >> 3 )
              {
LABEL_61:
                *(_DWORD *)(v2 + 2328) = v57;
                if ( *v52 != -8 )
                  --*(_DWORD *)(v2 + 2332);
                *v52 = v58;
                *((_DWORD *)v52 + 2) = 0;
                v52[2] = 0;
                goto LABEL_53;
              }
LABEL_66:
              sub_1E4AC20(v2 + 2312, v49);
              sub_1E48A60(v2 + 2312, &v67, &v68);
              v52 = v68;
              v58 = v67;
              v57 = *(_DWORD *)(v2 + 2328) + 1;
              goto LABEL_61;
            }
          }
          else
          {
            ++*(_QWORD *)(v2 + 2312);
          }
          v49 *= 2;
          goto LABEL_66;
        }
        v1 += 272;
        sub_1E0A0F0(*(_QWORD *)(v2 + 32), (__int64)v20, v18, v19, v22, v21);
        if ( v66 == v1 )
          return;
      }
    }
  }
}
