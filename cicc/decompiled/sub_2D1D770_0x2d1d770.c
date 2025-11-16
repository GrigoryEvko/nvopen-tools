// Function: sub_2D1D770
// Address: 0x2d1d770
//
_BOOL8 __fastcall sub_2D1D770(__int64 a1, _BYTE *a2, unsigned __int64 a3, __int64 a4, _QWORD *a5, _QWORD *a6)
{
  __int64 v9; // rsi
  _QWORD *v10; // rax
  __int64 v11; // rcx
  __int64 v12; // rdx
  unsigned __int64 v13; // rdx
  _QWORD *v14; // rax
  __int64 v15; // r8
  __int64 v16; // rsi
  __int64 v17; // rcx
  _BOOL4 v18; // r12d
  _BYTE *v20; // rbx
  unsigned __int64 v21; // r13
  __int64 v22; // rdi
  _QWORD *v23; // rax
  __int64 v24; // rsi
  unsigned __int8 *v25; // rsi
  __m128i v26; // xmm0
  __m128i v27; // xmm1
  __m128i v28; // xmm2
  unsigned __int64 *v29; // rdi
  __int64 *v30; // rax
  __int64 *v31; // rax
  char v32; // al
  bool v33; // al
  _QWORD *v34; // rax
  __int64 v35; // r8
  __int64 v36; // rcx
  __int64 v37; // rdx
  char v38; // [rsp+27h] [rbp-379h]
  _QWORD *v40; // [rsp+38h] [rbp-368h]
  unsigned __int64 v41; // [rsp+40h] [rbp-360h] BYREF
  _BYTE *v42; // [rsp+48h] [rbp-358h] BYREF
  __int64 v43; // [rsp+50h] [rbp-350h] BYREF
  unsigned __int64 i; // [rsp+58h] [rbp-348h] BYREF
  __m128i v45; // [rsp+60h] [rbp-340h] BYREF
  __m128i v46; // [rsp+70h] [rbp-330h] BYREF
  __m128i v47; // [rsp+80h] [rbp-320h] BYREF
  __m128i v48[3]; // [rsp+90h] [rbp-310h] BYREF
  char v49; // [rsp+C0h] [rbp-2E0h]
  unsigned __int64 *v50[2]; // [rsp+D0h] [rbp-2D0h] BYREF
  __int64 v51; // [rsp+E0h] [rbp-2C0h]
  __int64 v52; // [rsp+E8h] [rbp-2B8h] BYREF
  unsigned int v53; // [rsp+F0h] [rbp-2B0h]
  _QWORD v54[2]; // [rsp+228h] [rbp-178h] BYREF
  char v55; // [rsp+238h] [rbp-168h]
  _BYTE *v56; // [rsp+240h] [rbp-160h]
  __int64 v57; // [rsp+248h] [rbp-158h]
  _BYTE v58[128]; // [rsp+250h] [rbp-150h] BYREF
  __int16 v59; // [rsp+2D0h] [rbp-D0h]
  _QWORD v60[2]; // [rsp+2D8h] [rbp-C8h] BYREF
  __int64 v61; // [rsp+2E8h] [rbp-B8h]
  __int64 v62; // [rsp+2F0h] [rbp-B0h] BYREF
  unsigned int v63; // [rsp+2F8h] [rbp-A8h]
  char v64; // [rsp+370h] [rbp-30h] BYREF

  v42 = a2;
  v41 = a3;
  if ( *a2 == 61 )
  {
    sub_D665A0(&v45, (__int64)a2);
    v40 = a5 + 1;
    v9 = (__int64)(a5 + 1);
    v10 = (_QWORD *)a5[2];
    if ( v10 )
    {
      do
      {
        while ( 1 )
        {
          v11 = v10[2];
          v12 = v10[3];
          if ( v10[4] >= (unsigned __int64)v42 )
            break;
          v10 = (_QWORD *)v10[3];
          if ( !v12 )
            goto LABEL_7;
        }
        v9 = (__int64)v10;
        v10 = (_QWORD *)v10[2];
      }
      while ( v11 );
LABEL_7:
      if ( v40 != (_QWORD *)v9 && *(_QWORD *)(v9 + 32) <= (unsigned __int64)v42 )
      {
LABEL_10:
        v13 = *(_QWORD *)(v9 + 40);
        v43 = v13;
        if ( v13 )
        {
          v14 = (_QWORD *)a6[2];
          v15 = (__int64)(a6 + 1);
          if ( !v14 )
            goto LABEL_36;
          do
          {
            while ( 1 )
            {
              v16 = v14[2];
              v17 = v14[3];
              if ( v14[4] >= v13 )
                break;
              v14 = (_QWORD *)v14[3];
              if ( !v17 )
                goto LABEL_16;
            }
            v15 = (__int64)v14;
            v14 = (_QWORD *)v14[2];
          }
          while ( v16 );
LABEL_16:
          if ( a6 + 1 == (_QWORD *)v15 || *(_QWORD *)(v15 + 32) > v13 )
          {
LABEL_36:
            v50[0] = (unsigned __int64 *)&v43;
            v15 = sub_2D1BF10(a6, v15, v50);
          }
          a4 = *(_QWORD *)(v15 + 40);
          v43 = a4;
        }
        else
        {
          v43 = a4;
        }
        if ( !a4 || !(unsigned __int8)sub_2D1CFB0(a1 + 112, a4, v41) )
          return 1;
        v23 = (_QWORD *)a5[2];
        if ( v23 )
        {
          v24 = (__int64)(a5 + 1);
          do
          {
            if ( v23[4] < v41 )
            {
              v23 = (_QWORD *)v23[3];
            }
            else
            {
              v24 = (__int64)v23;
              v23 = (_QWORD *)v23[2];
            }
          }
          while ( v23 );
          if ( v40 != (_QWORD *)v24 && *(_QWORD *)(v24 + 32) <= v41 )
            goto LABEL_47;
        }
        else
        {
          v24 = (__int64)(a5 + 1);
        }
        v50[0] = &v41;
        v24 = sub_2D1BF10(a5, v24, v50);
LABEL_47:
        v25 = *(unsigned __int8 **)(v24 + 40);
        for ( i = (unsigned __int64)v25; ; i = (unsigned __int64)v25 )
        {
          v26 = _mm_loadu_si128(&v45);
          v49 = 1;
          v27 = _mm_loadu_si128(&v46);
          v28 = _mm_loadu_si128(&v47);
          v50[1] = 0;
          v29 = *(unsigned __int64 **)(a1 + 8);
          v30 = &v52;
          v51 = 1;
          v48[0] = v26;
          v50[0] = v29;
          v48[1] = v27;
          v48[2] = v28;
          do
          {
            *v30 = -4;
            v30 += 5;
            *(v30 - 4) = -3;
            *(v30 - 3) = -4;
            *(v30 - 2) = -3;
          }
          while ( v30 != v54 );
          v54[0] = v60;
          v54[1] = 0;
          v56 = v58;
          v57 = 0x400000000LL;
          v55 = 0;
          v60[0] = &unk_49DDBE8;
          v60[1] = 0;
          v61 = 1;
          v59 = 256;
          v31 = &v62;
          do
          {
            *v31 = -4096;
            v31 += 2;
          }
          while ( v31 != (__int64 *)&v64 );
          v32 = sub_CF63E0(v29, v25, v48, (__int64)v50);
          v60[0] = &unk_49DDBE8;
          v38 = v32;
          if ( (v61 & 1) == 0 )
            sub_C7D6A0(v62, 16LL * v63, 8);
          nullsub_184();
          if ( v56 != v58 )
            _libc_free((unsigned __int64)v56);
          if ( (v51 & 1) == 0 )
            sub_C7D6A0(v52, 40LL * v53, 8);
          v33 = sub_CEA640(i);
          if ( (v38 & 2) != 0 && !v33 )
            break;
          if ( i == v43 )
            return 1;
          v34 = (_QWORD *)a5[2];
          v35 = (__int64)(a5 + 1);
          if ( !v34 )
            goto LABEL_68;
          do
          {
            while ( 1 )
            {
              v36 = v34[2];
              v37 = v34[3];
              if ( v34[4] >= i )
                break;
              v34 = (_QWORD *)v34[3];
              if ( !v37 )
                goto LABEL_66;
            }
            v35 = (__int64)v34;
            v34 = (_QWORD *)v34[2];
          }
          while ( v36 );
LABEL_66:
          if ( v40 == (_QWORD *)v35 || *(_QWORD *)(v35 + 32) > i )
          {
LABEL_68:
            v50[0] = &i;
            v35 = sub_2D1BF10(a5, v35, v50);
          }
          v25 = *(unsigned __int8 **)(v35 + 40);
        }
        return 0;
      }
    }
    else
    {
      v9 = (__int64)(a5 + 1);
    }
    v50[0] = (unsigned __int64 *)&v42;
    v9 = sub_2D1BF10(a5, v9, v50);
    goto LABEL_10;
  }
  v18 = sub_CEA680((__int64)a2);
  if ( !v18 )
    return 1;
  v20 = v42;
  if ( v42 )
    v20 = v42 + 24;
  v21 = v41 + 24;
  if ( !v41 )
    v21 = 0;
  while ( (_BYTE *)v21 != v20 )
  {
    v22 = (__int64)(v20 - 24);
    if ( !v20 )
      v22 = 0;
    if ( (unsigned __int8)sub_B46490(v22) )
      return 0;
    v20 = (_BYTE *)*((_QWORD *)v20 + 1);
  }
  return v18;
}
