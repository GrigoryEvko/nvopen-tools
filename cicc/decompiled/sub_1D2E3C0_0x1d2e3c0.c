// Function: sub_1D2E3C0
// Address: 0x1d2e3c0
//
__int64 __fastcall sub_1D2E3C0(
        __int64 a1,
        __int64 a2,
        unsigned __int16 a3,
        __int64 a4,
        int a5,
        __int64 a6,
        __int64 *a7,
        __int64 a8)
{
  bool v9; // zf
  __int64 *v11; // r13
  __int64 *i; // r14
  __int64 v13; // rsi
  __int64 v14; // rsi
  _QWORD *v15; // r13
  __int64 v16; // rsi
  __int64 v17; // r12
  int v18; // r9d
  __int64 *v19; // r13
  __int16 v20; // dx
  __int64 *v21; // r14
  unsigned __int64 **v22; // rcx
  __int64 v23; // rsi
  __int64 v24; // rax
  __int64 *v25; // rdx
  __int64 *v26; // rax
  unsigned __int64 v27; // r14
  unsigned int v28; // eax
  unsigned __int64 v29; // rdx
  __int64 v30; // rcx
  __int64 v31; // rdx
  __int64 *v32; // rdx
  __int64 *v34; // r8
  _QWORD *v35; // rax
  _BYTE *v36; // r9
  __int64 v37; // r14
  _QWORD *v38; // r15
  _BYTE **v39; // r8
  _QWORD *v40; // rax
  __int64 v41; // rax
  __int64 v42; // r8
  unsigned int v43; // eax
  _QWORD *v44; // rdx
  _QWORD *v45; // rsi
  unsigned int v48; // [rsp+28h] [rbp-198h]
  unsigned __int64 **v50; // [rsp+30h] [rbp-190h]
  _BYTE *v51; // [rsp+30h] [rbp-190h]
  __int64 v52; // [rsp+30h] [rbp-190h]
  _BYTE **v53; // [rsp+38h] [rbp-188h]
  __int64 *v54; // [rsp+48h] [rbp-178h] BYREF
  _BYTE *v55; // [rsp+50h] [rbp-170h] BYREF
  __int64 v56; // [rsp+58h] [rbp-168h]
  _BYTE v57[128]; // [rsp+60h] [rbp-160h] BYREF
  unsigned __int64 *v58; // [rsp+E0h] [rbp-E0h] BYREF
  __int64 v59; // [rsp+E8h] [rbp-D8h]
  _BYTE *v60; // [rsp+F0h] [rbp-D0h] BYREF
  __int64 v61; // [rsp+F8h] [rbp-C8h]
  int v62; // [rsp+100h] [rbp-C0h]
  _BYTE v63[184]; // [rsp+108h] [rbp-B8h] BYREF

  v9 = *(_BYTE *)(a4 + 16LL * (unsigned int)(a5 - 1)) == 111;
  v54 = 0;
  if ( !v9 )
  {
    v58 = (unsigned __int64 *)&v60;
    v59 = 0x2000000000LL;
    sub_16BD430((__int64)&v58, a3);
    sub_16BD4C0((__int64)&v58, a4);
    v11 = &a7[2 * a8];
    for ( i = a7; v11 != i; sub_16BD430((__int64)&v58, *((_DWORD *)i - 2)) )
    {
      v13 = *i;
      i += 2;
      sub_16BD4C0((__int64)&v58, v13);
    }
    v14 = *(_QWORD *)(a2 + 72);
    v55 = (_BYTE *)v14;
    if ( v14 )
      sub_1623A60((__int64)&v55, v14, 2);
    LODWORD(v56) = *(_DWORD *)(a2 + 64);
    v15 = sub_1D17920(a1, (__int64)&v58, (__int64)&v55, (__int64 *)&v54);
    if ( v55 )
      sub_161E7C0((__int64)&v55, (__int64)v55);
    if ( v15 )
    {
      v16 = *(_QWORD *)(a2 + 72);
      v55 = (_BYTE *)v16;
      if ( v16 )
        sub_1623A60((__int64)&v55, v16, 2);
      LODWORD(v56) = *(_DWORD *)(a2 + 64);
      v17 = sub_1D18310(a1, (__int64)v15, (__int64)&v55);
      if ( v55 )
        sub_161E7C0((__int64)&v55, (__int64)v55);
      if ( v58 != (unsigned __int64 *)&v60 )
        _libc_free((unsigned __int64)v58);
      return v17;
    }
    if ( v58 != (unsigned __int64 *)&v60 )
      _libc_free((unsigned __int64)v58);
  }
  if ( !(unsigned __int8)sub_1D2D480(a1, a2, a3) )
    v54 = 0;
  v19 = *(__int64 **)(a2 + 32);
  v58 = 0;
  v61 = 16;
  *(_WORD *)(a2 + 24) = a3;
  v20 = a3;
  v62 = 0;
  *(_QWORD *)(a2 + 40) = a4;
  *(_DWORD *)(a2 + 60) = a5;
  v59 = (__int64)v63;
  v60 = v63;
  v21 = &v19[5 * *(unsigned int *)(a2 + 56)];
  if ( v21 != v19 )
  {
    v22 = &v58;
    while ( 1 )
    {
      v23 = *v19;
      v19 += 5;
      if ( v23 )
      {
        v24 = *(v19 - 1);
        *(_QWORD *)*(v19 - 2) = v24;
        if ( v24 )
          *(_QWORD *)(v24 + 24) = *(v19 - 2);
      }
      *(v19 - 5) = 0;
      *((_DWORD *)v19 - 8) = 0;
      v25 = *(__int64 **)(v23 + 48);
      if ( v25 )
        goto LABEL_21;
      v26 = (__int64 *)v59;
      if ( v60 == (_BYTE *)v59 )
      {
        v34 = (__int64 *)(v59 + 8LL * HIDWORD(v61));
        v18 = HIDWORD(v61);
        if ( (__int64 *)v59 == v34 )
        {
LABEL_83:
          if ( HIDWORD(v61) >= (unsigned int)v61 )
            goto LABEL_27;
          v18 = ++HIDWORD(v61);
          *v34 = v23;
          v58 = (unsigned __int64 *)((char *)v58 + 1);
        }
        else
        {
          while ( v23 != *v26 )
          {
            if ( *v26 == -2 )
              v25 = v26;
            if ( v34 == ++v26 )
            {
              if ( !v25 )
                goto LABEL_83;
              *v25 = v23;
              --v62;
              v58 = (unsigned __int64 *)((char *)v58 + 1);
              break;
            }
          }
        }
LABEL_21:
        if ( v19 == v21 )
          goto LABEL_28;
      }
      else
      {
LABEL_27:
        v50 = v22;
        sub_16CCBA0((__int64)v22, v23);
        v22 = v50;
        if ( v19 == v21 )
        {
LABEL_28:
          v20 = *(_WORD *)(a2 + 24);
          v19 = *(__int64 **)(a2 + 32);
          break;
        }
      }
    }
  }
  if ( v20 < 0 )
  {
    *(_QWORD *)(a2 + 88) = 0;
    *(_QWORD *)(a2 + 96) = 0;
  }
  if ( v19 )
  {
    v27 = *(unsigned int *)(a2 + 56);
    v28 = 0;
    if ( *(_DWORD *)(a2 + 56) )
    {
      if ( --v27 )
      {
        _BitScanReverse64(&v27, v27);
        v28 = 64 - (v27 ^ 0x3F);
        v27 = 8LL * v28;
      }
    }
    v29 = *(unsigned int *)(a1 + 472);
    if ( (unsigned int)v29 <= v28 )
    {
      v43 = v28 + 1;
      v42 = v43;
      if ( v43 >= v29 )
      {
        if ( v43 > v29 )
        {
          if ( v43 > (unsigned __int64)*(unsigned int *)(a1 + 476) )
          {
            v48 = v43;
            v52 = v43;
            sub_16CD150(a1 + 464, (const void *)(a1 + 480), v43, 8, v43, v18);
            v29 = *(unsigned int *)(a1 + 472);
            v43 = v48;
            v42 = v52;
          }
          v30 = *(_QWORD *)(a1 + 464);
          v44 = (_QWORD *)(v30 + 8 * v29);
          v45 = (_QWORD *)(v30 + 8 * v42);
          if ( v44 != v45 )
          {
            do
            {
              if ( v44 )
                *v44 = 0;
              ++v44;
            }
            while ( v45 != v44 );
            v30 = *(_QWORD *)(a1 + 464);
          }
          *(_DWORD *)(a1 + 472) = v43;
          goto LABEL_37;
        }
      }
      else
      {
        *(_DWORD *)(a1 + 472) = v43;
      }
    }
    v30 = *(_QWORD *)(a1 + 464);
LABEL_37:
    *v19 = *(_QWORD *)(v30 + v27);
    *(_QWORD *)(*(_QWORD *)(a1 + 464) + v27) = v19;
    *(_DWORD *)(a2 + 56) = 0;
    *(_QWORD *)(a2 + 32) = 0;
  }
  sub_1D23B60(a1, a2, (__int64)a7, a8);
  v31 = HIDWORD(v61);
  if ( HIDWORD(v61) != v62 )
  {
    v56 = 0x1000000000LL;
    v35 = v60;
    v55 = v57;
    if ( v60 != (_BYTE *)v59 )
      v31 = (unsigned int)v61;
    v36 = &v60[8 * v31];
    if ( v60 == v36 )
      goto LABEL_57;
    while ( 1 )
    {
      v37 = *v35;
      v38 = v35;
      if ( *v35 < 0xFFFFFFFFFFFFFFFELL )
        break;
      if ( v36 == (_BYTE *)++v35 )
        goto LABEL_57;
    }
    if ( v36 == (_BYTE *)v35 )
    {
LABEL_57:
      v39 = &v55;
    }
    else
    {
      v39 = &v55;
      if ( !*(_QWORD *)(v37 + 48) )
        goto LABEL_69;
      while ( 1 )
      {
        v40 = v38 + 1;
        if ( v38 + 1 == (_QWORD *)v36 )
          break;
        v37 = *v40;
        for ( ++v38; *v40 >= 0xFFFFFFFFFFFFFFFELL; v38 = v40 )
        {
          if ( v36 == (_BYTE *)++v40 )
            goto LABEL_58;
          v37 = *v40;
        }
        if ( v36 == (_BYTE *)v38 )
          break;
        if ( !*(_QWORD *)(v37 + 48) )
        {
LABEL_69:
          v41 = (unsigned int)v56;
          if ( (unsigned int)v56 >= HIDWORD(v56) )
          {
            v51 = v36;
            v53 = v39;
            sub_16CD150((__int64)v39, v57, 0, 8, (int)v39, (int)v36);
            v41 = (unsigned int)v56;
            v36 = v51;
            v39 = v53;
          }
          *(_QWORD *)&v55[8 * v41] = v37;
          LODWORD(v56) = v56 + 1;
        }
      }
    }
LABEL_58:
    sub_1D2D860(a1, (__int64)v39);
    if ( v55 != v57 )
    {
      _libc_free((unsigned __int64)v55);
      v32 = v54;
      if ( !v54 )
        goto LABEL_41;
      goto LABEL_40;
    }
  }
  v32 = v54;
  if ( v54 )
LABEL_40:
    sub_16BDA20((__int64 *)(a1 + 320), (__int64 *)a2, v32);
LABEL_41:
  if ( v60 != (_BYTE *)v59 )
    _libc_free((unsigned __int64)v60);
  return a2;
}
