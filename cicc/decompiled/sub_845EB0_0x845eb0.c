// Function: sub_845EB0
// Address: 0x845eb0
//
__int64 __fastcall sub_845EB0(
        const __m128i **a1,
        __m128i *a2,
        __int64 a3,
        _BYTE *a4,
        __int16 a5,
        unsigned int a6,
        _DWORD *a7)
{
  __int64 v7; // r11
  _BYTE *v8; // r15
  char v10; // al
  unsigned int v11; // ebx
  char v12; // al
  int v13; // edx
  const __m128i *v14; // r14
  __int64 v16; // rax
  int v17; // ecx
  const __m128i *v18; // rdi
  __int8 v19; // dl
  const __m128i *v20; // rax
  const __m128i *v21; // rax
  __int8 i; // dl
  __int64 v23; // rdx
  __int64 v24; // rcx
  __int64 v25; // r8
  __int64 v26; // r9
  int v27; // ebx
  int v28; // eax
  __int8 v29; // cl
  const __m128i *v30; // rax
  __int8 v31; // dl
  __int8 v32; // dl
  __m128i *v33; // rax
  char v34; // al
  __m128i *v35; // rax
  __int64 v36; // rcx
  __int64 v37; // r8
  int v38; // eax
  unsigned int v39; // [rsp+4h] [rbp-1DCh]
  int v40; // [rsp+8h] [rbp-1D8h]
  __int64 v42; // [rsp+18h] [rbp-1C8h] BYREF
  _BYTE v43[48]; // [rsp+20h] [rbp-1C0h] BYREF
  _OWORD v44[9]; // [rsp+50h] [rbp-190h] BYREF
  __m128i v45; // [rsp+E0h] [rbp-100h]
  __m128i v46; // [rsp+F0h] [rbp-F0h]
  __m128i v47; // [rsp+100h] [rbp-E0h]
  __m128i v48; // [rsp+110h] [rbp-D0h]
  __m128i v49; // [rsp+120h] [rbp-C0h]
  __m128i v50; // [rsp+130h] [rbp-B0h]
  __m128i v51; // [rsp+140h] [rbp-A0h]
  __m128i v52; // [rsp+150h] [rbp-90h]
  __m128i v53; // [rsp+160h] [rbp-80h]
  __m128i v54; // [rsp+170h] [rbp-70h]
  __m128i v55; // [rsp+180h] [rbp-60h]
  __m128i v56; // [rsp+190h] [rbp-50h]
  __m128i v57; // [rsp+1A0h] [rbp-40h]

  v7 = a3;
  v8 = a4;
  *a7 = 0;
  v44[0] = _mm_loadu_si128((const __m128i *)a1);
  v44[1] = _mm_loadu_si128((const __m128i *)a1 + 1);
  v44[2] = _mm_loadu_si128((const __m128i *)a1 + 2);
  v10 = *((_BYTE *)a1 + 16);
  v44[3] = _mm_loadu_si128((const __m128i *)a1 + 3);
  v44[4] = _mm_loadu_si128((const __m128i *)a1 + 4);
  v44[5] = _mm_loadu_si128((const __m128i *)a1 + 5);
  v44[6] = _mm_loadu_si128((const __m128i *)a1 + 6);
  v44[7] = _mm_loadu_si128((const __m128i *)a1 + 7);
  v44[8] = _mm_loadu_si128((const __m128i *)a1 + 8);
  if ( v10 == 2 )
  {
    v45 = _mm_loadu_si128((const __m128i *)a1 + 9);
    v46 = _mm_loadu_si128((const __m128i *)a1 + 10);
    v47 = _mm_loadu_si128((const __m128i *)a1 + 11);
    v48 = _mm_loadu_si128((const __m128i *)a1 + 12);
    v49 = _mm_loadu_si128((const __m128i *)a1 + 13);
    v50 = _mm_loadu_si128((const __m128i *)a1 + 14);
    v51 = _mm_loadu_si128((const __m128i *)a1 + 15);
    v52 = _mm_loadu_si128((const __m128i *)a1 + 16);
    v53 = _mm_loadu_si128((const __m128i *)a1 + 17);
    v54 = _mm_loadu_si128((const __m128i *)a1 + 18);
    v55 = _mm_loadu_si128((const __m128i *)a1 + 19);
    v56 = _mm_loadu_si128((const __m128i *)a1 + 20);
    v57 = _mm_loadu_si128((const __m128i *)a1 + 21);
    goto LABEL_4;
  }
  if ( v10 != 5 && v10 != 1 )
  {
LABEL_4:
    v11 = a5 & 0xC00;
    if ( a4 )
      goto LABEL_5;
LABEL_17:
    v40 = 0;
    if ( v11 )
      v11 = 2048;
LABEL_19:
    v8 = v43;
    if ( !(unsigned int)sub_840D60(
                          (__m128i *)a1,
                          a2,
                          0,
                          v7,
                          1u,
                          1u,
                          v7,
                          v11,
                          a6,
                          (FILE *)((char *)a1 + 68),
                          (__int64)v43,
                          0) )
    {
      *a7 = 1;
      return sub_6E4BC0((__int64)a1, (__int64)v44);
    }
    goto LABEL_27;
  }
  v11 = a5 & 0xC00;
  v45.m128i_i64[0] = (__int64)a1[18];
  if ( !a4 )
    goto LABEL_17;
LABEL_5:
  v12 = a4[17];
  if ( (v12 & 2) != 0 )
  {
    a4[17] = v12 & 0xFD;
    v16 = *(_QWORD *)a4;
    if ( *(_QWORD *)a4
      && *(_BYTE *)(v16 + 174) == 3
      && ((v39 = a6, v35 = sub_73D790(*(_QWORD *)(v16 + 152)), v7 = a3, a6 = v39, v35 == a2)
       || (v38 = sub_8D97D0(v35, a2, 0, v36, v37), v7 = a3, a6 = v39, v38)) )
    {
      v8[17] |= 2u;
      v40 = 0;
    }
    else
    {
      v40 = 1;
    }
  }
  else
  {
    v40 = 0;
  }
  if ( v11 )
    v11 = 2048;
  if ( (v8[16] & 0x88) != 0 )
    goto LABEL_19;
  if ( *(_QWORD *)v8 )
    goto LABEL_11;
  sub_6F69D0(a1, 8u);
LABEL_27:
  if ( !*(_QWORD *)v8 )
  {
    if ( (v8[16] & 0x10) != 0 )
      goto LABEL_13;
LABEL_29:
    v14 = a2;
    sub_845370((__m128i *)a1, a2, (__int64)v8);
    goto LABEL_30;
  }
LABEL_11:
  if ( (unsigned int)sub_8D97B0(a2) )
  {
    *a7 = 1;
    v14 = a2;
    sub_6E6260(a1);
    goto LABEL_30;
  }
  if ( (v8[16] & 0x10) == 0 )
    goto LABEL_29;
LABEL_13:
  sub_8449E0(a1, 0, (__int64)v8, 0, 0);
  v13 = dword_4F077C4;
  if ( (dword_4F077C4 != 2 || unk_4F07778 <= 201102 && !dword_4F07774) && !dword_4F07734 )
  {
    v14 = *a1;
    goto LABEL_30;
  }
  v14 = *a1;
  if ( ((*a1)[8].m128i_i8[12] & 0xFB) == 8 )
  {
    v27 = sub_8D4C10(*a1, dword_4F077C4 != 2) & 0xFFFFFF8F;
    if ( (a2[8].m128i_i8[12] & 0xFB) != 8 )
    {
      v28 = 0;
LABEL_57:
      v14 = *a1;
      if ( v28 != v27 && *((_BYTE *)a1 + 16) )
      {
        v29 = v14[8].m128i_i8[12];
        if ( v29 == 12 )
        {
          v30 = *a1;
          do
          {
            v30 = (const __m128i *)v30[10].m128i_i64[0];
            v31 = v30[8].m128i_i8[12];
          }
          while ( v31 == 12 );
        }
        else
        {
          v31 = v14[8].m128i_i8[12];
        }
        if ( v31 )
        {
          v32 = a2[8].m128i_i8[12];
          if ( v32 == 12 )
          {
            v33 = a2;
            do
            {
              v33 = (__m128i *)v33[10].m128i_i64[0];
              v32 = v33[8].m128i_i8[12];
            }
            while ( v32 == 12 );
          }
          if ( v32 )
          {
            if ( v29 == 12 )
            {
              do
                v14 = (const __m128i *)v14[10].m128i_i64[0];
              while ( v14[8].m128i_i8[12] == 12 );
            }
            v14 = sub_73CA70(v14, (__int64)a2);
          }
        }
      }
      goto LABEL_30;
    }
    v13 = dword_4F077C4;
LABEL_56:
    v28 = sub_8D4C10(a2, v13 != 2) & 0xFFFFFF8F;
    goto LABEL_57;
  }
  if ( (a2[8].m128i_i8[12] & 0xFB) == 8 )
  {
    v27 = 0;
    goto LABEL_56;
  }
LABEL_30:
  v17 = sub_8319F0((__int64)a1, 0);
  if ( !*((_BYTE *)a1 + 16) )
    goto LABEL_35;
  v18 = *a1;
  v19 = (*a1)[8].m128i_i8[12];
  if ( v19 == 12 )
  {
    v20 = *a1;
    do
    {
      v20 = (const __m128i *)v20[10].m128i_i64[0];
      v19 = v20[8].m128i_i8[12];
    }
    while ( v19 == 12 );
  }
  if ( !v19 )
    goto LABEL_35;
  if ( !v17 )
    goto LABEL_70;
  if ( *((_BYTE *)a1 + 17) == 1 )
  {
    if ( v18 == v14 )
      goto LABEL_35;
    goto LABEL_50;
  }
  if ( !(unsigned int)sub_8D3A70(v18) || *((_BYTE *)a1 + 17) != 2 )
  {
LABEL_70:
    if ( (v8[16] & 0x10) != 0 )
    {
      v34 = *((_BYTE *)a1 + 17);
      if ( v34 == 1 )
      {
        sub_6F7690((const __m128i *)a1, (__int64)v14);
      }
      else if ( v34 == 2 )
      {
        sub_6F7980((__m128i *)a1, (__int64)v14);
      }
    }
    sub_8443E0((__m128i *)a1, (__int64)v14, 1);
    goto LABEL_35;
  }
  sub_6FA340((__int64)a1, 0, v23, v24, v25, v26);
  if ( *a1 == v14 || !(unsigned int)sub_8319F0((__int64)a1, 0) )
    goto LABEL_35;
LABEL_50:
  a1[18]->m128i_i64[0] = (__int64)v14;
  *a1 = v14;
LABEL_35:
  if ( v40 )
  {
    if ( (unsigned int)sub_8319F0((__int64)a1, &v42) )
    {
      *(_BYTE *)(*(_QWORD *)(v42 + 56) + 50LL) |= 0x10u;
      sub_6F7690((const __m128i *)a1, (__int64)a2);
      return sub_6E4BC0((__int64)a1, (__int64)v44);
    }
    if ( *((_BYTE *)a1 + 16) )
    {
      v21 = *a1;
      for ( i = (*a1)[8].m128i_i8[12]; i == 12; i = v21[8].m128i_i8[12] )
        v21 = (const __m128i *)v21[10].m128i_i64[0];
      if ( i )
        sub_721090();
    }
    sub_6E6870((__int64)a1);
  }
  sub_6F7690((const __m128i *)a1, (__int64)a2);
  return sub_6E4BC0((__int64)a1, (__int64)v44);
}
