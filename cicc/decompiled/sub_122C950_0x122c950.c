// Function: sub_122C950
// Address: 0x122c950
//
__int64 __fastcall sub_122C950(_QWORD **a1, __int64 *a2, char a3)
{
  __int64 v3; // r14
  int v5; // eax
  char v6; // al
  unsigned int v7; // r14d
  _QWORD *v8; // r15
  _QWORD *v9; // r10
  __int64 v10; // rcx
  __int64 v11; // r8
  __int64 *v12; // rdi
  __int64 v13; // rax
  int v15; // eax
  __int64 v16; // rax
  __int64 v17; // rax
  _QWORD *v18; // rax
  __int64 v19; // rax
  __int64 v20; // rax
  _QWORD *v21; // rax
  __int64 v22; // rax
  __int64 v23; // rax
  __int64 v24; // r15
  __int64 v25; // rax
  __int64 v26; // rax
  _QWORD *v27; // [rsp+8h] [rbp-1A8h]
  __int64 v28; // [rsp+10h] [rbp-1A0h]
  _QWORD *v29; // [rsp+10h] [rbp-1A0h]
  __int64 v30; // [rsp+18h] [rbp-198h]
  __int64 v31; // [rsp+18h] [rbp-198h]
  __int64 v32; // [rsp+18h] [rbp-198h]
  _QWORD v33[4]; // [rsp+20h] [rbp-190h] BYREF
  __int16 v34; // [rsp+40h] [rbp-170h]
  _QWORD v35[4]; // [rsp+50h] [rbp-160h] BYREF
  __int16 v36; // [rsp+70h] [rbp-140h]
  __m128i v37; // [rsp+80h] [rbp-130h] BYREF
  __int64 v38; // [rsp+90h] [rbp-120h]
  __int64 v39; // [rsp+98h] [rbp-118h]
  _QWORD *v40; // [rsp+A0h] [rbp-110h]
  __int64 v41; // [rsp+A8h] [rbp-108h]
  __int64 v42; // [rsp+B0h] [rbp-100h]
  __m128i v43; // [rsp+C0h] [rbp-F0h] BYREF
  unsigned __int64 v44; // [rsp+D0h] [rbp-E0h]
  __int64 v45; // [rsp+D8h] [rbp-D8h]
  _QWORD *v46; // [rsp+E0h] [rbp-D0h]
  __int64 v47; // [rsp+E8h] [rbp-C8h]
  __int64 v48; // [rsp+F0h] [rbp-C0h]
  __m128i v49; // [rsp+100h] [rbp-B0h] BYREF
  unsigned __int64 v50; // [rsp+110h] [rbp-A0h]
  __int64 v51; // [rsp+118h] [rbp-98h]
  __int64 v52; // [rsp+120h] [rbp-90h]
  __int64 v53; // [rsp+128h] [rbp-88h]
  __int64 v54; // [rsp+130h] [rbp-80h]
  __m128i v55; // [rsp+140h] [rbp-70h] BYREF
  unsigned __int64 v56; // [rsp+150h] [rbp-60h]
  __int64 v57; // [rsp+158h] [rbp-58h]
  __int64 v58; // [rsp+160h] [rbp-50h]
  __int64 v59; // [rsp+168h] [rbp-48h]
  __int64 v60; // [rsp+170h] [rbp-40h]

  v3 = (__int64)(a1 + 22);
  v44 = 0x8000000000000000LL;
  v50 = 0x8000000000000000LL;
  v56 = 0x8000000000000000LL;
  v39 = 0x7FFFFFFFFFFFFFFFLL;
  v45 = 0x7FFFFFFFFFFFFFFFLL;
  v51 = 0x7FFFFFFFFFFFFFFFLL;
  v57 = 0x7FFFFFFFFFFFFFFFLL;
  v37 = (__m128i)0xFFFFFFFFFFFFFFFFLL;
  v38 = -1;
  v40 = 0;
  v41 = 0;
  v42 = 0;
  v43 = 0u;
  v46 = 0;
  v47 = 256;
  v48 = 0;
  v49 = 0u;
  v52 = 0;
  v53 = 256;
  v54 = 0;
  v55 = 0u;
  v58 = 0;
  v59 = 256;
  v60 = 0;
  *((_DWORD *)a1 + 60) = sub_1205200((__int64)(a1 + 22));
  if ( (unsigned __int8)sub_120AFE0((__int64)a1, 12, "expected '(' here") )
    return 1;
  v5 = *((_DWORD *)a1 + 60);
  if ( v5 != 13 )
  {
    if ( v5 == 507 )
    {
      do
      {
        if ( (unsigned int)sub_2241AC0(a1 + 31, "count") )
        {
          if ( (unsigned int)sub_2241AC0(a1 + 31, "lowerBound") )
          {
            if ( (unsigned int)sub_2241AC0(a1 + 31, "upperBound") )
            {
              if ( (unsigned int)sub_2241AC0(a1 + 31, "stride") )
              {
                v33[2] = a1 + 31;
                v33[0] = "invalid field '";
                v35[0] = v33;
                v34 = 1027;
                v35[2] = "'";
                v36 = 770;
                goto LABEL_26;
              }
              v6 = sub_122C010((__int64)a1, (__int64)"stride", 6, &v55);
            }
            else
            {
              v6 = sub_122C010((__int64)a1, (__int64)"upperBound", 10, &v49);
            }
          }
          else
          {
            v6 = sub_122C010((__int64)a1, (__int64)"lowerBound", 10, &v43);
          }
        }
        else
        {
          v6 = sub_122C010((__int64)a1, (__int64)"count", 5, &v37);
        }
        if ( v6 )
          return 1;
        if ( *((_DWORD *)a1 + 60) != 4 )
          goto LABEL_8;
        v15 = sub_1205200(v3);
        *((_DWORD *)a1 + 60) = v15;
      }
      while ( v15 == 507 );
    }
    v35[0] = "expected field label here";
    v36 = 259;
LABEL_26:
    sub_11FD800(v3, (unsigned __int64)a1[29], (__int64)v35, 1);
    return 1;
  }
LABEL_8:
  v7 = sub_120AFE0((__int64)a1, 13, "expected ')' here");
  if ( (_BYTE)v7 )
    return 1;
  if ( HIDWORD(v42) == 1 )
  {
    v24 = v37.m128i_i64[0];
    v25 = sub_BCB2E0(*a1);
    v26 = sub_ACD640(v25, v24, 1u);
    v8 = sub_B98A20(v26, v24);
  }
  else
  {
    v8 = 0;
    if ( HIDWORD(v42) == 2 )
      v8 = v40;
  }
  if ( HIDWORD(v48) == 1 )
  {
    v32 = v43.m128i_i64[0];
    v22 = sub_BCB2E0(*a1);
    v23 = sub_ACD640(v22, v32, 1u);
    v9 = sub_B98A20(v23, v32);
  }
  else
  {
    v9 = 0;
    if ( HIDWORD(v48) == 2 )
      v9 = v46;
  }
  if ( HIDWORD(v54) == 1 )
  {
    v29 = v9;
    v31 = v49.m128i_i64[0];
    v19 = sub_BCB2E0(*a1);
    v20 = sub_ACD640(v19, v31, 1u);
    v21 = sub_B98A20(v20, v31);
    v9 = v29;
    v10 = (__int64)v21;
  }
  else
  {
    v10 = 0;
    if ( HIDWORD(v54) == 2 )
      v10 = v52;
  }
  if ( HIDWORD(v60) == 1 )
  {
    v27 = v9;
    v28 = v10;
    v30 = v55.m128i_i64[0];
    v16 = sub_BCB2E0(*a1);
    v17 = sub_ACD640(v16, v30, 1u);
    v18 = sub_B98A20(v17, v30);
    v10 = v28;
    v9 = v27;
    v11 = (__int64)v18;
  }
  else
  {
    v11 = 0;
    if ( HIDWORD(v60) == 2 )
      v11 = v58;
  }
  v12 = *a1;
  if ( a3 )
    v13 = sub_B02F70(v12, (__int64)v8, (__int64)v9, v10, v11, 1u, 1);
  else
    v13 = sub_B02F70(v12, (__int64)v8, (__int64)v9, v10, v11, 0, 1);
  *a2 = v13;
  return v7;
}
