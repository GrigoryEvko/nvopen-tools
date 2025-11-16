// Function: sub_36F1310
// Address: 0x36f1310
//
void __fastcall sub_36F1310(__int64 a1, unsigned __int64 a2, __int64 a3)
{
  __int64 v4; // r12
  unsigned __int16 v5; // ax
  __int64 *v6; // rax
  __int64 v7; // r12
  __int64 *v8; // r12
  __int64 v9; // rax
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 v12; // r15
  __int64 v13; // rbx
  __int64 i; // r15
  char *v15; // r14
  char v16; // al
  __int64 v17; // rax
  __int64 v18; // rax
  unsigned __int64 v19; // rdx
  char **v20; // rax
  unsigned int v21; // edx
  __int64 v22; // rdx
  _QWORD *v23; // rdi
  _BYTE *v24; // r9
  unsigned __int64 v25; // r8
  unsigned __int64 v26; // rsi
  unsigned __int64 j; // rcx
  unsigned __int64 v28; // rax
  __int16 v29; // bx
  __int64 v30; // rsi
  unsigned __int16 v31; // ax
  unsigned __int64 v32; // rdx
  __int16 v33; // cx
  __int16 v34; // dx
  unsigned __int64 v35; // rdi
  unsigned __int64 *v36; // rbx
  unsigned __int64 v37; // r12
  unsigned __int64 v38; // rdi
  unsigned __int8 v39; // [rsp+Fh] [rbp-F1h]
  __int64 v40; // [rsp+10h] [rbp-F0h]
  unsigned int v41; // [rsp+10h] [rbp-F0h]
  __int64 v42; // [rsp+18h] [rbp-E8h]
  __int64 *v43; // [rsp+20h] [rbp-E0h] BYREF
  unsigned int v44; // [rsp+28h] [rbp-D8h]
  __m128i v45; // [rsp+30h] [rbp-D0h] BYREF
  _BYTE *v46; // [rsp+40h] [rbp-C0h] BYREF
  __int64 v47; // [rsp+48h] [rbp-B8h]
  _BYTE v48[48]; // [rsp+50h] [rbp-B0h] BYREF
  __int64 v49; // [rsp+80h] [rbp-80h] BYREF
  __int64 v50; // [rsp+88h] [rbp-78h]
  __int64 *v51; // [rsp+90h] [rbp-70h]
  __int64 *v52; // [rsp+98h] [rbp-68h]
  __int64 v53; // [rsp+A0h] [rbp-60h]
  unsigned __int64 *v54; // [rsp+A8h] [rbp-58h]
  __int64 *v55; // [rsp+B0h] [rbp-50h]
  __int64 v56; // [rsp+B8h] [rbp-48h]
  __int64 v57; // [rsp+C0h] [rbp-40h]
  __int64 *v58; // [rsp+C8h] [rbp-38h]

  v4 = *(_QWORD *)(a1 + 24);
  v40 = sub_B2BD20(a1);
  v42 = sub_B2BEC0(v4);
  v39 = sub_303E610(a3, v4, v40, v42);
  v5 = sub_B2BD00(a1);
  if ( !HIBYTE(v5) )
    LOBYTE(v5) = 0;
  if ( v39 > (unsigned __int8)v5 )
  {
    v6 = (__int64 *)sub_B2BE50(v4);
    v7 = sub_A77A40(v6, v39);
    sub_B2D5C0(a1, 86);
    sub_B2D460(a1, v7);
    v49 = 0;
    v46 = v48;
    v47 = 0x300000000LL;
    v51 = 0;
    v52 = 0;
    v53 = 0;
    v54 = 0;
    v55 = 0;
    v56 = 0;
    v57 = 0;
    v58 = 0;
    v50 = 8;
    v49 = sub_22077B0(0x40u);
    v8 = (__int64 *)(v49 + ((4 * v50 - 4) & 0xFFFFFFFFFFFFFFF8LL));
    v9 = sub_22077B0(0x200u);
    v54 = (unsigned __int64 *)v8;
    *v8 = v9;
    v58 = v8;
    v52 = (__int64 *)v9;
    v53 = v9 + 512;
    v56 = v9;
    v57 = v9 + 512;
    v51 = (__int64 *)v9;
    v55 = (__int64 *)v9;
    v45 = (__m128i)a2;
    sub_36F10E0((unsigned __int64 *)&v49, &v45);
    while ( v51 != v55 )
    {
      v12 = *v51;
      v13 = v51[1];
      if ( v51 == (__int64 *)(v53 - 16) )
      {
        j_j___libc_free_0((unsigned __int64)v52);
        v22 = *++v54 + 512;
        v52 = (__int64 *)*v54;
        v53 = v22;
        v51 = v52;
      }
      else
      {
        v51 += 2;
      }
      for ( i = *(_QWORD *)(v12 + 16); i; i = *(_QWORD *)(i + 8) )
      {
        v15 = *(char **)(i + 24);
        v16 = *v15;
        if ( (unsigned __int8)*v15 > 0x1Cu )
        {
          if ( v16 == 61 )
          {
            v18 = (unsigned int)v47;
            v19 = (unsigned int)v47 + 1LL;
            if ( v19 > HIDWORD(v47) )
            {
              sub_C8D5F0((__int64)&v46, v48, v19, 0x10u, v10, v11);
              v18 = (unsigned int)v47;
            }
            v20 = (char **)&v46[16 * v18];
            *v20 = v15;
            v20[1] = (char *)v13;
            LODWORD(v47) = v47 + 1;
          }
          else if ( (unsigned __int8)(v16 - 78) <= 1u )
          {
            v45.m128i_i64[0] = *(_QWORD *)(i + 24);
            v45.m128i_i64[1] = v13;
            sub_36F10E0((unsigned __int64 *)&v49, &v45);
          }
          else if ( v16 == 63 )
          {
            v44 = sub_AE2980(v42, 0x65u)[3];
            if ( v44 > 0x40 )
              sub_C43690((__int64)&v43, 0, 0);
            else
              v43 = 0;
            if ( (unsigned __int8)sub_B4DE60((__int64)v15, v42, (__int64)&v43) )
            {
              v41 = v44;
              if ( v44 > 0x40 )
              {
                v21 = v41 - sub_C444A0((__int64)&v43);
                v17 = -1;
                if ( v21 <= 0x40 )
                  v17 = *v43;
              }
              else
              {
                v17 = (__int64)v43;
              }
              v45.m128i_i64[0] = (__int64)v15;
              v45.m128i_i64[1] = v13 + v17;
              sub_36F10E0((unsigned __int64 *)&v49, &v45);
            }
            if ( v44 > 0x40 )
            {
              if ( v43 )
                j_j___libc_free_0_0((unsigned __int64)v43);
            }
          }
        }
      }
    }
    v23 = v46;
    v24 = &v46[16 * (unsigned int)v47];
    v25 = 1LL << v39;
    if ( v24 == v46 )
    {
LABEL_42:
      v35 = v49;
      if ( v49 )
      {
        v36 = v54;
        v37 = (unsigned __int64)(v58 + 1);
        if ( v58 + 1 > (__int64 *)v54 )
        {
          do
          {
            v38 = *v36++;
            j_j___libc_free_0(v38);
          }
          while ( v37 > (unsigned __int64)v36 );
          v35 = v49;
        }
        j_j___libc_free_0(v35);
      }
      if ( v46 != v48 )
        _libc_free((unsigned __int64)v46);
      return;
    }
    while ( 1 )
    {
      v26 = v23[1];
      if ( v25 )
        break;
      if ( v26 )
        goto LABEL_39;
      v30 = *v23;
      v34 = 510;
      v31 = *(_WORD *)(*v23 + 2LL);
LABEL_41:
      v23 += 2;
      *(_WORD *)(v30 + 2) = v34 | v31 & 0xFF81;
      if ( v23 == (_QWORD *)v24 )
        goto LABEL_42;
    }
    if ( v26 )
    {
      for ( j = v25 % v26; j; j = v28 % j )
      {
        v28 = v26;
        v26 = j;
      }
    }
    else
    {
      v26 = 1LL << v39;
    }
LABEL_39:
    _BitScanReverse64(&v26, v26);
    v29 = 63 - (v26 ^ 0x3F);
    v30 = *v23;
    v31 = *(_WORD *)(*v23 + 2LL);
    _BitScanReverse64(&v32, 1LL << (v31 >> 1));
    v33 = 63 - (v32 ^ 0x3F);
    v34 = 2 * v29;
    if ( (unsigned __int8)v33 > (unsigned __int8)v29 )
      v34 = 2 * v33;
    goto LABEL_41;
  }
}
