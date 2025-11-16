// Function: sub_1922AA0
// Address: 0x1922aa0
//
void __fastcall sub_1922AA0(
        __int64 a1,
        __int64 a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  __int64 *v10; // r13
  __int64 v11; // r15
  __int64 *v14; // r14
  double v15; // xmm4_8
  double v16; // xmm5_8
  _QWORD *v17; // rsi
  __int64 *v18; // r8
  __int64 *v19; // rax
  __int64 *v20; // rdi
  __int64 *v21; // r15
  __int64 *v22; // rax
  __int64 v23; // r13
  __int64 *v24; // r14
  __int64 v25; // rdx
  _QWORD *v26; // rax
  _QWORD *v27; // rsi
  signed __int64 v28; // rdx
  _QWORD *v29; // rdx
  __int64 *v30; // rax
  __int64 v31; // [rsp+10h] [rbp-80h] BYREF
  __int64 *v32; // [rsp+18h] [rbp-78h]
  __int64 *v33; // [rsp+20h] [rbp-70h]
  __int64 v34; // [rsp+28h] [rbp-68h]
  int v35; // [rsp+30h] [rbp-60h]
  _BYTE v36[88]; // [rsp+38h] [rbp-58h] BYREF

  v10 = (__int64 *)v36;
  v11 = *(_QWORD *)(a2 + 8);
  v31 = 0;
  v32 = (__int64 *)v36;
  v33 = (__int64 *)v36;
  v34 = 4;
  v35 = 0;
  if ( !v11 )
    goto LABEL_21;
  v14 = (__int64 *)v36;
  do
  {
LABEL_5:
    v17 = sub_1648700(v11);
    if ( *((_BYTE *)v17 + 16) == 23 )
    {
      if ( v14 != v10 )
        goto LABEL_3;
      v18 = &v14[HIDWORD(v34)];
      if ( v18 == v14 )
      {
LABEL_56:
        if ( HIDWORD(v34) >= (unsigned int)v34 )
        {
LABEL_3:
          sub_16CCBA0((__int64)&v31, (__int64)v17);
          v14 = v33;
          v10 = v32;
          goto LABEL_4;
        }
        ++HIDWORD(v34);
        *v18 = (__int64)v17;
        v10 = v32;
        ++v31;
        v14 = v33;
      }
      else
      {
        v19 = v14;
        v20 = 0;
        while ( v17 != (_QWORD *)*v19 )
        {
          if ( *v19 == -2 )
            v20 = v19;
          if ( v18 == ++v19 )
          {
            if ( !v20 )
              goto LABEL_56;
            *v20 = (__int64)v17;
            v14 = v33;
            --v35;
            v11 = *(_QWORD *)(v11 + 8);
            ++v31;
            v10 = v32;
            if ( v11 )
              goto LABEL_5;
            goto LABEL_15;
          }
        }
      }
    }
LABEL_4:
    v11 = *(_QWORD *)(v11 + 8);
  }
  while ( v11 );
LABEL_15:
  if ( v10 == v14 )
    v21 = &v14[HIDWORD(v34)];
  else
    v21 = &v14[(unsigned int)v34];
  if ( v21 != v14 )
  {
    v22 = v14;
    while ( 1 )
    {
      v23 = *v22;
      v24 = v22;
      if ( (unsigned __int64)*v22 < 0xFFFFFFFFFFFFFFFELL )
        break;
      if ( v21 == ++v22 )
        goto LABEL_21;
    }
    if ( v21 != v22 )
    {
      do
      {
        v25 = 24LL * (*(_DWORD *)(v23 + 20) & 0xFFFFFFF);
        if ( (*(_BYTE *)(v23 + 23) & 0x40) != 0 )
        {
          v26 = *(_QWORD **)(v23 - 8);
          v27 = &v26[(unsigned __int64)v25 / 8];
        }
        else
        {
          v27 = (_QWORD *)v23;
          v26 = (_QWORD *)(v23 - v25);
        }
        v28 = 0xAAAAAAAAAAAAAAABLL * (v25 >> 3);
        if ( v28 >> 2 )
        {
          v29 = &v26[12 * (v28 >> 2)];
          while ( *v26 == a2 )
          {
            if ( v26[3] != a2 )
            {
              v26 += 3;
              break;
            }
            if ( v26[6] != a2 )
            {
              v26 += 6;
              break;
            }
            if ( v26[9] != a2 )
            {
              v26 += 9;
              break;
            }
            v26 += 12;
            if ( v26 == v29 )
            {
              v28 = 0xAAAAAAAAAAAAAAABLL * (v27 - v26);
              goto LABEL_43;
            }
          }
LABEL_34:
          if ( v27 != v26 )
            goto LABEL_35;
          goto LABEL_46;
        }
LABEL_43:
        if ( v28 != 2 )
        {
          if ( v28 != 3 )
          {
            if ( v28 != 1 )
              goto LABEL_46;
            goto LABEL_51;
          }
          if ( *v26 != a2 )
            goto LABEL_34;
          v26 += 3;
        }
        if ( *v26 != a2 )
          goto LABEL_34;
        v26 += 3;
LABEL_51:
        if ( *v26 != a2 )
          goto LABEL_34;
LABEL_46:
        sub_164D160(v23, a2, a3, a4, a5, a6, v15, v16, a9, a10);
        sub_386B550(*(_QWORD *)(a1 + 256), v23);
LABEL_35:
        v30 = v24 + 1;
        if ( v24 + 1 == v21 )
          break;
        v23 = *v30;
        for ( ++v24; (unsigned __int64)*v30 >= 0xFFFFFFFFFFFFFFFELL; v24 = v30 )
        {
          if ( v21 == ++v30 )
            goto LABEL_21;
          v23 = *v30;
        }
      }
      while ( v21 != v24 );
    }
  }
LABEL_21:
  if ( v33 != v32 )
    _libc_free((unsigned __int64)v33);
}
