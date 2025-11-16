// Function: sub_164C7D0
// Address: 0x164c7d0
//
void __fastcall sub_164C7D0(
        __int64 a1,
        __int64 a2,
        unsigned __int8 (__fastcall *a3)(__int64, __int64 *),
        __int64 a4,
        __m128 a5,
        double a6,
        double a7,
        double a8,
        double a9,
        double a10,
        double a11,
        __m128 a12)
{
  __int64 *v15; // r15
  __int64 *v16; // r13
  _QWORD *v17; // rsi
  __int64 v18; // rsi
  unsigned __int64 v19; // rax
  __int64 v20; // rax
  unsigned int i; // eax
  __int64 *v22; // rcx
  __int64 *v23; // rsi
  __int64 v24; // rax
  __int64 *v25; // rdi
  __int64 v26; // rax
  __int64 *v27; // rbx
  __int64 *v28; // r12
  __int64 v29; // rax
  _QWORD *v30; // rax
  char v31; // dl
  unsigned int v32; // eax
  unsigned __int64 *v33; // rdi
  __int64 v34; // rax
  bool v35; // zf
  _QWORD *v36; // r8
  _QWORD *v37; // rdi
  __int64 v38; // [rsp+10h] [rbp-1A0h]
  unsigned __int64 v40; // [rsp+20h] [rbp-190h] BYREF
  __int64 v41; // [rsp+28h] [rbp-188h]
  _QWORD *v42; // [rsp+30h] [rbp-180h]
  __int64 v43; // [rsp+40h] [rbp-170h] BYREF
  _BYTE *v44; // [rsp+48h] [rbp-168h]
  _BYTE *v45; // [rsp+50h] [rbp-160h]
  __int64 v46; // [rsp+58h] [rbp-158h]
  int v47; // [rsp+60h] [rbp-150h]
  _BYTE v48[72]; // [rsp+68h] [rbp-148h] BYREF
  __int64 *v49; // [rsp+B0h] [rbp-100h] BYREF
  __int64 v50; // [rsp+B8h] [rbp-F8h]
  _BYTE v51[240]; // [rsp+C0h] [rbp-F0h] BYREF

  v49 = (__int64 *)v51;
  v50 = 0x800000000LL;
  v43 = 0;
  v44 = v48;
  v45 = v48;
  v46 = 8;
  v47 = 0;
  if ( (*(_BYTE *)(a1 + 23) & 0x10) != 0 )
  {
    sub_16303F0(a1, a2, a5, a6, a7, a8, a9, a10, a11, a12);
    v15 = *(__int64 **)(a1 + 8);
    if ( v15 )
      goto LABEL_3;
LABEL_14:
    for ( i = v50; (_DWORD)v50; i = v50 )
    {
      v22 = v49;
      v40 = 6;
      v41 = 0;
      v23 = &v49[3 * i - 3];
      v42 = (_QWORD *)v23[2];
      if ( v42 != 0 && v42 + 1 != 0 && v42 != (_QWORD *)-16LL )
      {
        sub_1649AC0(&v40, *v23 & 0xFFFFFFFFFFFFFFF8LL);
        v22 = v49;
        i = v50;
      }
      v24 = i - 1;
      LODWORD(v50) = v24;
      v25 = &v22[3 * v24];
      v26 = v25[2];
      LOBYTE(v22) = v26 != -8;
      if ( ((v26 != 0) & (unsigned __int8)v22) != 0 && v26 != -16 )
        sub_1649B30(v25);
      sub_15A5060((__int64)v42, (_BYTE *)a1, a2, v22, *(double *)a5.m128_u64, a6, a7);
      if ( v42 + 1 != 0 && v42 != 0 && v42 != (_QWORD *)-16LL )
        sub_1649B30(&v40);
    }
    if ( v45 != v44 )
      _libc_free((unsigned __int64)v45);
  }
  else
  {
    v15 = *(__int64 **)(a1 + 8);
    if ( v15 )
    {
LABEL_3:
      v38 = a2 + 8;
      while ( 1 )
      {
        v16 = v15;
        v15 = (__int64 *)v15[1];
        if ( a3(a4, v16) )
        {
          v17 = sub_1648700((__int64)v16);
          if ( (unsigned __int8)(*((_BYTE *)v17 + 16) - 4) <= 0xCu )
          {
            v30 = v44;
            if ( v45 == v44 )
            {
              v36 = &v44[8 * HIDWORD(v46)];
              if ( v44 != (_BYTE *)v36 )
              {
                v37 = 0;
                while ( v17 != (_QWORD *)*v30 )
                {
                  if ( *v30 == -2 )
                    v37 = v30;
                  if ( v36 == ++v30 )
                  {
                    if ( !v37 )
                      goto LABEL_61;
                    *v37 = v17;
                    --v47;
                    ++v43;
                    goto LABEL_38;
                  }
                }
                goto LABEL_13;
              }
LABEL_61:
              if ( HIDWORD(v46) < (unsigned int)v46 )
              {
                ++HIDWORD(v46);
                *v36 = v17;
                ++v43;
LABEL_38:
                v40 = 6;
                v41 = 0;
                v42 = v17;
                if ( v17 != (_QWORD *)-8LL && v17 != (_QWORD *)-16LL )
                  sub_164C220((__int64)&v40);
                v32 = v50;
                if ( (unsigned int)v50 >= HIDWORD(v50) )
                {
                  sub_164AD50((__int64)&v49, 0);
                  v32 = v50;
                }
                v33 = (unsigned __int64 *)&v49[3 * v32];
                if ( v33 )
                {
                  *v33 = 6;
                  v33[1] = 0;
                  v34 = (__int64)v42;
                  v35 = v42 + 1 == 0;
                  v33[2] = (unsigned __int64)v42;
                  if ( v34 != 0 && !v35 && v34 != -16 )
                    sub_1649AC0(v33, v40 & 0xFFFFFFFFFFFFFFF8LL);
                  v32 = v50;
                }
                LODWORD(v50) = v32 + 1;
                if ( v42 + 1 != 0 && v42 != 0 && v42 != (_QWORD *)-16LL )
                  sub_1649B30(&v40);
                goto LABEL_13;
              }
            }
            sub_16CCBA0(&v43, v17);
            if ( v31 )
              goto LABEL_38;
          }
          else
          {
            if ( *v16 )
            {
              v18 = v16[1];
              v19 = v16[2] & 0xFFFFFFFFFFFFFFFCLL;
              *(_QWORD *)v19 = v18;
              if ( v18 )
                *(_QWORD *)(v18 + 16) = *(_QWORD *)(v18 + 16) & 3LL | v19;
            }
            *v16 = a2;
            if ( a2 )
            {
              v20 = *(_QWORD *)(a2 + 8);
              v16[1] = v20;
              if ( v20 )
                *(_QWORD *)(v20 + 16) = (unsigned __int64)(v16 + 1) | *(_QWORD *)(v20 + 16) & 3LL;
              v16[2] = v38 | v16[2] & 3;
              *(_QWORD *)(a2 + 8) = v16;
            }
          }
        }
LABEL_13:
        if ( !v15 )
          goto LABEL_14;
      }
    }
  }
  v27 = v49;
  v28 = &v49[3 * (unsigned int)v50];
  if ( v49 != v28 )
  {
    do
    {
      v29 = *(v28 - 1);
      v28 -= 3;
      if ( v29 != -8 && v29 != 0 && v29 != -16 )
        sub_1649B30(v28);
    }
    while ( v27 != v28 );
    v28 = v49;
  }
  if ( v28 != (__int64 *)v51 )
    _libc_free((unsigned __int64)v28);
}
