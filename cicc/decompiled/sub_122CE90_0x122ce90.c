// Function: sub_122CE90
// Address: 0x122ce90
//
__int64 __fastcall sub_122CE90(__int64 a1, __int64 *a2, char a3)
{
  __int64 v3; // r13
  int v6; // eax
  char v7; // al
  unsigned int v8; // r13d
  __int64 v9; // r14
  __int64 v10; // r15
  __int64 v11; // r10
  __int64 v12; // r8
  int v14; // eax
  __int64 *v15; // rdi
  __int64 v16; // rax
  __int64 *v17; // rdi
  __int64 *v18; // rdi
  __int64 *v19; // rdi
  __int64 v20; // [rsp+0h] [rbp-1A0h]
  _QWORD v21[4]; // [rsp+10h] [rbp-190h] BYREF
  __int16 v22; // [rsp+30h] [rbp-170h]
  __int64 v23; // [rsp+40h] [rbp-160h] BYREF
  __int64 v24; // [rsp+48h] [rbp-158h]
  char *v25; // [rsp+50h] [rbp-150h]
  __int16 v26; // [rsp+60h] [rbp-140h]
  __m128i v27; // [rsp+70h] [rbp-130h] BYREF
  unsigned __int64 v28; // [rsp+80h] [rbp-120h]
  __int64 v29; // [rsp+88h] [rbp-118h]
  __int64 v30; // [rsp+90h] [rbp-110h]
  __int64 v31; // [rsp+98h] [rbp-108h]
  __int64 v32; // [rsp+A0h] [rbp-100h]
  __m128i v33; // [rsp+B0h] [rbp-F0h] BYREF
  unsigned __int64 v34; // [rsp+C0h] [rbp-E0h]
  __int64 v35; // [rsp+C8h] [rbp-D8h]
  __int64 v36; // [rsp+D0h] [rbp-D0h]
  __int64 v37; // [rsp+D8h] [rbp-C8h]
  __int64 v38; // [rsp+E0h] [rbp-C0h]
  __m128i v39; // [rsp+F0h] [rbp-B0h] BYREF
  unsigned __int64 v40; // [rsp+100h] [rbp-A0h]
  __int64 v41; // [rsp+108h] [rbp-98h]
  __int64 v42; // [rsp+110h] [rbp-90h]
  __int64 v43; // [rsp+118h] [rbp-88h]
  __int64 v44; // [rsp+120h] [rbp-80h]
  __m128i v45; // [rsp+130h] [rbp-70h] BYREF
  unsigned __int64 v46; // [rsp+140h] [rbp-60h]
  __int64 v47; // [rsp+148h] [rbp-58h]
  __int64 v48; // [rsp+150h] [rbp-50h]
  __int64 v49; // [rsp+158h] [rbp-48h]
  __int64 v50; // [rsp+160h] [rbp-40h]

  v3 = a1 + 176;
  v28 = 0x8000000000000000LL;
  v34 = 0x8000000000000000LL;
  v40 = 0x8000000000000000LL;
  v46 = 0x8000000000000000LL;
  v29 = 0x7FFFFFFFFFFFFFFFLL;
  v35 = 0x7FFFFFFFFFFFFFFFLL;
  v41 = 0x7FFFFFFFFFFFFFFFLL;
  v27 = 0u;
  v30 = 0;
  v31 = 256;
  v32 = 0;
  v33 = 0u;
  v36 = 0;
  v37 = 256;
  v38 = 0;
  v39 = 0u;
  v42 = 0;
  v43 = 256;
  v44 = 0;
  v45 = 0u;
  v47 = 0x7FFFFFFFFFFFFFFFLL;
  v48 = 0;
  v49 = 256;
  v50 = 0;
  *(_DWORD *)(a1 + 240) = sub_1205200(a1 + 176);
  if ( (unsigned __int8)sub_120AFE0(a1, 12, "expected '(' here") )
    return 1;
  v6 = *(_DWORD *)(a1 + 240);
  if ( v6 != 13 )
  {
    if ( v6 == 507 )
    {
      do
      {
        if ( (unsigned int)sub_2241AC0(a1 + 248, "count") )
        {
          if ( (unsigned int)sub_2241AC0(a1 + 248, "lowerBound") )
          {
            if ( (unsigned int)sub_2241AC0(a1 + 248, "upperBound") )
            {
              if ( (unsigned int)sub_2241AC0(a1 + 248, "stride") )
              {
                v21[2] = a1 + 248;
                v21[0] = "invalid field '";
                v23 = (__int64)v21;
                v22 = 1027;
                v25 = "'";
                v26 = 770;
                goto LABEL_24;
              }
              v7 = sub_122C010(a1, (__int64)"stride", 6, &v45);
            }
            else
            {
              v7 = sub_122C010(a1, (__int64)"upperBound", 10, &v39);
            }
          }
          else
          {
            v7 = sub_122C010(a1, (__int64)"lowerBound", 10, &v33);
          }
        }
        else
        {
          v7 = sub_122C010(a1, (__int64)"count", 5, &v27);
        }
        if ( v7 )
          return 1;
        if ( *(_DWORD *)(a1 + 240) != 4 )
          goto LABEL_8;
        v14 = sub_1205200(v3);
        *(_DWORD *)(a1 + 240) = v14;
      }
      while ( v14 == 507 );
    }
    v23 = (__int64)"expected field label here";
    v26 = 259;
LABEL_24:
    sub_11FD800(v3, *(_QWORD *)(a1 + 232), (__int64)&v23, 1);
    return 1;
  }
LABEL_8:
  v8 = sub_120AFE0(a1, 13, "expected ')' here");
  if ( (_BYTE)v8 )
    return 1;
  if ( HIDWORD(v32) == 1 )
  {
    v19 = *(__int64 **)a1;
    v23 = 17;
    v24 = v27.m128i_i64[0];
    v9 = sub_B0D000(v19, &v23, 2, 0, 1);
  }
  else
  {
    v9 = 0;
    if ( HIDWORD(v32) == 2 )
      v9 = v30;
  }
  if ( HIDWORD(v38) == 1 )
  {
    v18 = *(__int64 **)a1;
    v23 = 17;
    v24 = v33.m128i_i64[0];
    v10 = sub_B0D000(v18, &v23, 2, 0, 1);
  }
  else
  {
    v10 = 0;
    if ( HIDWORD(v38) == 2 )
      v10 = v36;
  }
  if ( HIDWORD(v44) == 1 )
  {
    v17 = *(__int64 **)a1;
    v23 = 17;
    v24 = v39.m128i_i64[0];
    v11 = sub_B0D000(v17, &v23, 2, 0, 1);
  }
  else
  {
    v11 = 0;
    if ( HIDWORD(v44) == 2 )
      v11 = v42;
  }
  if ( HIDWORD(v50) == 1 )
  {
    v15 = *(__int64 **)a1;
    v20 = v11;
    v23 = 17;
    v24 = v45.m128i_i64[0];
    v16 = sub_B0D000(v15, &v23, 2, 0, 1);
    v11 = v20;
    v12 = v16;
  }
  else
  {
    v12 = 0;
    if ( HIDWORD(v50) == 2 )
      v12 = v48;
  }
  *a2 = sub_B036F0(*(__int64 **)a1, v9, v10, v11, v12, a3 != 0, 1);
  return v8;
}
