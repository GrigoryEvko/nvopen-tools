// Function: sub_1868550
// Address: 0x1868550
//
__int64 __fastcall sub_1868550(
        __int64 a1,
        _BYTE *a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10,
        __int64 a11,
        __int64 a12,
        int a13,
        int a14)
{
  unsigned int v14; // r14d
  unsigned __int64 v15; // rbx
  _BYTE *v17; // rdx
  _BYTE *v18; // rax
  _BYTE *i; // rdx
  __int64 v20; // r14
  int v21; // r15d
  unsigned __int64 v22; // rax
  char v23; // dl
  _BYTE *v24; // rdi
  __int64 v26; // rbx
  __int64 v27; // r15
  double v28; // xmm4_8
  double v29; // xmm5_8
  __int64 v30; // rsi
  unsigned __int64 v31; // rax
  unsigned __int64 v32; // rbx
  unsigned __int64 v33; // rbx
  _BYTE *v34; // rcx
  __int64 v35; // r8
  __int64 v36; // rax
  _BYTE *v37; // rdx
  __int64 v38; // [rsp+8h] [rbp-148h]
  _BYTE *v39; // [rsp+10h] [rbp-140h] BYREF
  __int64 v40; // [rsp+18h] [rbp-138h]
  _BYTE v41[304]; // [rsp+20h] [rbp-130h] BYREF

  v14 = 0;
  v15 = *(_QWORD *)(a1 + 96);
  if ( v15 && *(_QWORD *)(a1 + 8) )
  {
    v40 = 0x1000000000LL;
    v17 = v41;
    v18 = v41;
    v39 = v41;
    if ( v15 > 0x10 )
    {
      a2 = v41;
      sub_16CD150((__int64)&v39, v41, v15, 16, a13, a14);
      v17 = v39;
      v18 = &v39[16 * (unsigned int)v40];
    }
    for ( i = &v17[16 * v15]; i != v18; v18 += 16 )
    {
      if ( v18 )
      {
        *(_QWORD *)v18 = 0;
        v18[8] = 0;
      }
    }
    v20 = *(_QWORD *)(a1 + 8);
    LODWORD(v40) = v15;
    v21 = 0;
    while ( v20 )
    {
      v22 = (unsigned __int64)sub_1648700(v20);
      v23 = *(_BYTE *)(v22 + 16);
      if ( v23 != 4 )
      {
        if ( v23 == 78 )
        {
          v31 = v22 | 4;
        }
        else
        {
          if ( v23 != 29 )
            goto LABEL_13;
          v31 = v22 & 0xFFFFFFFFFFFFFFFBLL;
        }
        v32 = v31 & 0xFFFFFFFFFFFFFFF8LL;
        if ( (v31 & 4) != 0 )
        {
          if ( v20 != v32 - 24 )
            goto LABEL_13;
        }
        else if ( v20 != v32 - 72 )
        {
LABEL_13:
          v24 = v39;
          goto LABEL_14;
        }
        v33 = v32 - 24LL * (*(_DWORD *)(v32 + 20) & 0xFFFFFFF);
        if ( (*(_BYTE *)(a1 + 18) & 1) != 0 )
          sub_15E08E0(a1, (__int64)a2);
        v34 = *(_BYTE **)(a1 + 88);
        if ( (_DWORD)v40 )
        {
          v35 = 16LL * (unsigned int)v40;
          v36 = 0;
          do
          {
            v24 = v39;
            v37 = &v39[v36];
            if ( !v39[v36 + 8] )
            {
              a2 = *(_BYTE **)v33;
              if ( *(_BYTE *)(*(_QWORD *)v33 + 16LL) > 0x10u )
              {
LABEL_40:
                if ( a2 != v34 )
                {
                  if ( ++v21 == (_DWORD)v40 )
                    goto LABEL_14;
                  v37[8] = 1;
                }
                goto LABEL_43;
              }
              if ( *(_QWORD *)v37 )
              {
                if ( *(_BYTE **)v37 != a2 )
                  goto LABEL_40;
              }
              else
              {
                *(_QWORD *)v37 = a2;
              }
            }
LABEL_43:
            v36 += 16;
            v33 += 24LL;
            v34 += 40;
          }
          while ( v35 != v36 );
        }
      }
      v20 = *(_QWORD *)(v20 + 8);
    }
    if ( (*(_BYTE *)(a1 + 18) & 1) != 0 )
      sub_15E08E0(a1, (__int64)a2);
    v26 = *(_QWORD *)(a1 + 88);
    v24 = v39;
    if ( (_DWORD)v40 )
    {
      v27 = 0;
      v14 = 0;
      v38 = 16LL * (unsigned int)v40;
      do
      {
        if ( !v24[v27 + 8] && *(_QWORD *)(v26 + 8) )
        {
          if ( !(unsigned __int8)sub_15E0470(v26)
            && (!(unsigned __int8)sub_15E0450(v26)
             || (unsigned __int8)sub_1560180(a1 + 112, 36)
             || (unsigned __int8)sub_1560180(a1 + 112, 37)) )
          {
            v30 = *(_QWORD *)&v39[v27];
            if ( !v30 )
              v30 = sub_1599EF0(*(__int64 ***)v26);
            v14 = 1;
            sub_164D160(v26, v30, a3, a4, a5, a6, v28, v29, a9, a10);
            v24 = v39;
          }
          else
          {
            v24 = v39;
          }
        }
        v26 += 40;
        v27 += 16;
      }
      while ( v38 != v27 );
    }
    else
    {
LABEL_14:
      v14 = 0;
    }
    if ( v24 != v41 )
      _libc_free((unsigned __int64)v24);
  }
  return v14;
}
