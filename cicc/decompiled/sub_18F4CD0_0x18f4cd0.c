// Function: sub_18F4CD0
// Address: 0x18f4cd0
//
__int64 __fastcall sub_18F4CD0(__int64 a1, __int64 a2, _QWORD *a3)
{
  __int64 v3; // rbx
  __int64 v5; // rax
  __m128i v6; // kr00_16
  __m128i v7; // kr10_16
  __int64 v8; // rax
  char v9; // dl
  int v10; // eax
  __int64 v11; // r13
  __int64 v12; // rcx
  __int64 v13; // r14
  __int64 v14; // rbx
  __int64 v15; // r13
  char v16; // al
  __int64 v17; // rcx
  unsigned int v18; // r12d
  int v20; // r8d
  int v21; // r9d
  __m128i v22; // xmm4
  __m128i v23; // xmm5
  unsigned int v24; // edx
  __m128i v25; // xmm0
  __m128i v26; // xmm1
  __m128i v27; // xmm2
  __m128i v28; // xmm3
  __m128i v29; // xmm6
  __m128i v30; // xmm7
  __int64 v31; // r13
  _QWORD *v32; // rax
  char v33; // dl
  int v34; // r8d
  int v35; // r9d
  __int64 v36; // r14
  __int64 v37; // rax
  __int64 v38; // rsi
  __int64 *v39; // rax
  __int64 *v40; // rdi
  __int64 *v41; // rcx
  __int64 v42; // [rsp+10h] [rbp-1E0h]
  __int64 v43; // [rsp+18h] [rbp-1D8h]
  __int64 v44; // [rsp+20h] [rbp-1D0h]
  __int64 v45; // [rsp+28h] [rbp-1C8h]
  __int64 v47; // [rsp+40h] [rbp-1B0h]
  __m128i v48; // [rsp+60h] [rbp-190h] BYREF
  __m128i v49; // [rsp+70h] [rbp-180h] BYREF
  __int64 v50; // [rsp+80h] [rbp-170h]
  __m128i v51; // [rsp+90h] [rbp-160h] BYREF
  __m128i v52; // [rsp+A0h] [rbp-150h]
  __int64 v53; // [rsp+B0h] [rbp-140h]
  char v54; // [rsp+B8h] [rbp-138h]
  __int64 v55; // [rsp+C0h] [rbp-130h] BYREF
  __int64 *v56; // [rsp+C8h] [rbp-128h]
  __int64 *v57; // [rsp+D0h] [rbp-120h]
  __int64 v58; // [rsp+D8h] [rbp-118h]
  int v59; // [rsp+E0h] [rbp-110h]
  _BYTE v60[72]; // [rsp+E8h] [rbp-108h] BYREF
  _BYTE *v61; // [rsp+130h] [rbp-C0h] BYREF
  __int64 v62; // [rsp+138h] [rbp-B8h]
  _BYTE v63[176]; // [rsp+140h] [rbp-B0h] BYREF

  v61 = v63;
  v62 = 0x1000000000LL;
  v55 = 0;
  v56 = (__int64 *)v60;
  v57 = (__int64 *)v60;
  v58 = 8;
  v59 = 0;
  if ( !a1 )
    BUG();
  v3 = *(_QWORD *)(a2 + 40);
  v43 = *(_QWORD *)(a1 + 32);
  v5 = 0;
  if ( a2 )
    v5 = a2 + 24;
  v42 = v5;
  v44 = *(_QWORD *)(a1 + 40);
  switch ( *(_BYTE *)(a2 + 16) )
  {
    case '6':
      sub_141EB40(&v48, (__int64 *)a2);
      goto LABEL_33;
    case '7':
      sub_141EDF0(&v48, a2);
      v27 = _mm_loadu_si128(&v48);
      v28 = _mm_loadu_si128(&v49);
      v24 = HIDWORD(v62);
      v53 = v50;
      v8 = (unsigned int)v62;
      v51 = v27;
      v52 = v28;
      goto LABEL_30;
    case ':':
      sub_141F110(&v48, a2);
      v29 = _mm_loadu_si128(&v48);
      v30 = _mm_loadu_si128(&v49);
      v24 = HIDWORD(v62);
      v53 = v50;
      v8 = (unsigned int)v62;
      v51 = v29;
      v52 = v30;
      goto LABEL_30;
    case ';':
      sub_141F3C0(&v48, a2);
LABEL_33:
      v25 = _mm_loadu_si128(&v48);
      v26 = _mm_loadu_si128(&v49);
      v24 = HIDWORD(v62);
      v53 = v50;
      v8 = (unsigned int)v62;
      v51 = v25;
      v52 = v26;
      goto LABEL_30;
    case 'R':
      sub_141F0A0(&v48, a2);
      v22 = _mm_loadu_si128(&v48);
      v23 = _mm_loadu_si128(&v49);
      v24 = HIDWORD(v62);
      v53 = v50;
      v8 = (unsigned int)v62;
      v51 = v22;
      v52 = v23;
LABEL_30:
      v47 = v53;
      v6 = v51;
      v7 = v52;
      if ( (unsigned int)v8 >= v24 )
      {
        sub_16CD150((__int64)&v61, v63, 0, 8, v20, v21);
        v8 = (unsigned int)v62;
      }
      break;
    default:
      v6 = v51;
      v7 = v52;
      v47 = v53;
      v8 = 0;
      break;
  }
  *(_QWORD *)&v61[8 * v8] = v3;
  v9 = 1;
  v10 = v62 + 1;
  LODWORD(v62) = v10;
  if ( v10 )
  {
    while ( 1 )
    {
      v11 = v43;
      v12 = *(_QWORD *)&v61[8 * v10 - 8];
      LODWORD(v62) = v10 - 1;
      v45 = v12;
      if ( v12 != v44 )
        v11 = *(_QWORD *)(v12 + 48);
      v13 = v11;
      v14 = v12 + 40;
      if ( v9 )
        v14 = v42;
      if ( v11 != v14 )
        break;
LABEL_26:
      if ( v45 != v44 )
      {
        v31 = *(_QWORD *)(v45 + 8);
        if ( v31 )
        {
          while ( 1 )
          {
            v32 = sub_1648700(v31);
            if ( (unsigned __int8)(*((_BYTE *)v32 + 16) - 25) <= 9u )
              break;
            v31 = *(_QWORD *)(v31 + 8);
            if ( !v31 )
              goto LABEL_27;
          }
LABEL_46:
          v38 = v32[5];
          v39 = v56;
          if ( v57 != v56 )
            goto LABEL_40;
          v40 = &v56[HIDWORD(v58)];
          if ( v56 != v40 )
          {
            v41 = 0;
            do
            {
              if ( v38 == *v39 )
                goto LABEL_44;
              if ( *v39 == -2 )
                v41 = v39;
              ++v39;
            }
            while ( v40 != v39 );
            if ( v41 )
            {
              *v41 = v38;
              --v59;
              ++v55;
              goto LABEL_41;
            }
          }
          if ( HIDWORD(v58) < (unsigned int)v58 )
          {
            ++HIDWORD(v58);
            *v40 = v38;
            ++v55;
          }
          else
          {
LABEL_40:
            sub_16CCBA0((__int64)&v55, v38);
            if ( !v33 )
              goto LABEL_44;
          }
LABEL_41:
          v36 = sub_1648700(v31)[5];
          v37 = (unsigned int)v62;
          if ( (unsigned int)v62 >= HIDWORD(v62) )
          {
            sub_16CD150((__int64)&v61, v63, 0, 8, v34, v35);
            v37 = (unsigned int)v62;
          }
          *(_QWORD *)&v61[8 * v37] = v36;
          LODWORD(v62) = v62 + 1;
LABEL_44:
          while ( 1 )
          {
            v31 = *(_QWORD *)(v31 + 8);
            if ( !v31 )
              break;
            v32 = sub_1648700(v31);
            if ( (unsigned __int8)(*((_BYTE *)v32 + 16) - 25) <= 9u )
              goto LABEL_46;
          }
        }
      }
LABEL_27:
      v10 = v62;
      v9 = 0;
      if ( !(_DWORD)v62 )
        goto LABEL_28;
    }
    while ( 1 )
    {
      if ( v13 )
      {
        v15 = v13 - 24;
        v16 = sub_15F3040(v13 - 24);
        if ( a2 == v13 - 24 || !v16 )
          goto LABEL_13;
      }
      else
      {
        if ( !(unsigned __int8)sub_15F3040(0) )
          goto LABEL_13;
        v15 = 0;
      }
      v54 = 1;
      v51 = v6;
      v52 = v7;
      v53 = v47;
      if ( (sub_13575E0(a3, v15, &v51, v17) & 2) != 0 )
      {
        v18 = 0;
        goto LABEL_19;
      }
LABEL_13:
      v13 = *(_QWORD *)(v13 + 8);
      if ( v14 == v13 )
        goto LABEL_26;
    }
  }
LABEL_28:
  v18 = 1;
LABEL_19:
  if ( v57 != v56 )
    _libc_free((unsigned __int64)v57);
  if ( v61 != v63 )
    _libc_free((unsigned __int64)v61);
  return v18;
}
