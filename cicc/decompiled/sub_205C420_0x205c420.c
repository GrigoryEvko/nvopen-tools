// Function: sub_205C420
// Address: 0x205c420
//
__int64 __fastcall sub_205C420(
        __int64 a1,
        int a2,
        char a3,
        int a4,
        __int64 a5,
        __int64 a6,
        __m128i a7,
        double a8,
        __m128i a9,
        __int64 a10)
{
  _QWORD *v10; // r13
  _QWORD *v12; // rbx
  __int64 v13; // r14
  int v14; // eax
  __int64 v15; // rsi
  __int64 v16; // rax
  __int64 v17; // r8
  __int64 v18; // r9
  __m128i *v19; // rsi
  __int32 v20; // edx
  __m128i v21; // rax
  int v22; // r15d
  __int64 *v23; // rax
  __int64 v24; // rcx
  __int64 v25; // r12
  __int64 v26; // rax
  _QWORD *v27; // r14
  _QWORD *v28; // r13
  int i; // ebx
  __m128i *v30; // rsi
  __m128i v31; // rax
  int v32; // r12d
  int v33; // eax
  __int64 v34; // r8
  __int64 v35; // r12
  __int64 v36; // rax
  char v37; // di
  unsigned int v38; // eax
  __int64 v39; // rax
  int v40; // eax
  __int64 v41; // r12
  __int64 v42; // r15
  __m128i *v43; // rsi
  __int64 v45; // [rsp+8h] [rbp-F8h]
  __int64 v46; // [rsp+20h] [rbp-E0h]
  __int64 v47; // [rsp+28h] [rbp-D8h]
  __int64 v48; // [rsp+30h] [rbp-D0h]
  __int64 v49; // [rsp+30h] [rbp-D0h]
  __int64 v50; // [rsp+30h] [rbp-D0h]
  __int64 v51; // [rsp+38h] [rbp-C8h]
  unsigned __int8 v52; // [rsp+40h] [rbp-C0h]
  __int64 v53; // [rsp+40h] [rbp-C0h]
  int v54; // [rsp+40h] [rbp-C0h]
  __int64 v55; // [rsp+48h] [rbp-B8h]
  int v56; // [rsp+48h] [rbp-B8h]
  char v57; // [rsp+5Bh] [rbp-A5h] BYREF
  unsigned int v58; // [rsp+5Ch] [rbp-A4h] BYREF
  __m128i v59; // [rsp+60h] [rbp-A0h] BYREF
  __int64 v60; // [rsp+70h] [rbp-90h] BYREF
  __int64 v61; // [rsp+78h] [rbp-88h]
  _QWORD v62[2]; // [rsp+80h] [rbp-80h] BYREF
  __int64 v63; // [rsp+90h] [rbp-70h] BYREF
  __int64 v64; // [rsp+98h] [rbp-68h]
  __int64 v65; // [rsp+A0h] [rbp-60h] BYREF
  __int64 v66; // [rsp+A8h] [rbp-58h]
  __m128i v67; // [rsp+B0h] [rbp-50h] BYREF
  __int64 v68; // [rsp+C0h] [rbp-40h]
  __int64 v69; // [rsp+110h] [rbp+10h]

  v10 = (_QWORD *)a6;
  v12 = (_QWORD *)a1;
  v13 = a10;
  v46 = *(_QWORD *)(a6 + 16);
  v14 = *(_DWORD *)(a1 + 112);
  v15 = a2 | (unsigned int)(8 * v14);
  if ( a3 )
  {
    v15 = (a4 << 16) | (unsigned int)v15 | 0x80000000;
  }
  else if ( v14 )
  {
    v40 = **(_DWORD **)(a1 + 104);
    if ( v40 < 0 )
      v15 = ((*(unsigned __int16 *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a6 + 32) + 40LL) + 24LL)
                                                          + 16LL * (v40 & 0x7FFFFFFF))
                                              & 0xFFFFFFFFFFFFFFF8LL)
                                  + 24LL)
            + 1) << 16)
          | (unsigned int)v15;
  }
  v16 = sub_1D38BB0(a6, v15, a5, 5, 0, 1, a7, a8, a9, 0);
  v19 = *(__m128i **)(a10 + 8);
  v59.m128i_i64[0] = v16;
  v59.m128i_i32[2] = v20;
  if ( v19 == *(__m128i **)(a10 + 16) )
  {
    sub_1D4B0A0((const __m128i **)a10, v19, &v59);
  }
  else
  {
    if ( v19 )
    {
      *v19 = _mm_loadu_si128(&v59);
      v19 = *(__m128i **)(a10 + 8);
    }
    *(_QWORD *)(a10 + 8) = v19 + 1;
  }
  v21.m128i_i64[0] = *(unsigned int *)(a1 + 8);
  if ( a2 == 4 )
  {
    v41 = v21.m128i_u32[0];
    v42 = 0;
    if ( v21.m128i_i32[0] )
    {
      do
      {
        v21.m128i_i64[0] = (__int64)sub_1D2A660(
                                      v10,
                                      *(_DWORD *)(*(_QWORD *)(a1 + 104) + 4 * v42),
                                      *(unsigned __int8 *)(*(_QWORD *)(a1 + 80) + v42),
                                      0,
                                      v17,
                                      v18);
        v43 = *(__m128i **)(a10 + 8);
        v67 = v21;
        if ( v43 == *(__m128i **)(a10 + 16) )
        {
          v21.m128i_i64[0] = sub_1D4B3A0((const __m128i **)a10, v43, &v67);
        }
        else
        {
          if ( v43 )
          {
            *v43 = v21;
            v43 = *(__m128i **)(a10 + 8);
          }
          *(_QWORD *)(a10 + 8) = v43 + 1;
        }
        ++v42;
      }
      while ( v41 != v42 );
    }
  }
  else if ( v21.m128i_i32[0] )
  {
    v47 = *(unsigned int *)(a1 + 8);
    v22 = 0;
    v51 = 0;
    do
    {
      v55 = v10[6];
      v23 = (__int64 *)(*v12 + 16 * v51);
      v24 = *v23;
      v25 = v23[1];
      v26 = (unsigned __int8)*v23;
      v60 = v24;
      v61 = v25;
      if ( (_BYTE)v26 )
      {
        v56 = *(unsigned __int8 *)(v46 + v26 + 1040);
      }
      else
      {
        v53 = v24;
        if ( !sub_1F58D20((__int64)&v60) )
        {
          v48 = v53;
          v33 = sub_1F58D40((__int64)&v60);
          v62[1] = v25;
          v54 = v33;
          v62[0] = v48;
          if ( sub_1F58D20((__int64)v62) )
          {
            v67.m128i_i8[0] = 0;
            v67.m128i_i64[1] = 0;
            LOBYTE(v63) = 0;
            sub_1F426C0(v46, v55, LODWORD(v62[0]), v25, (__int64)&v67, (unsigned int *)&v65, &v63);
            v37 = v63;
          }
          else
          {
            v34 = v25;
            v35 = v46;
            sub_1F40D10((__int64)&v67, v46, v55, v48, v34);
            v36 = v67.m128i_u8[8];
            LOBYTE(v63) = v67.m128i_i8[8];
            v64 = v68;
            if ( v67.m128i_i8[8] )
              goto LABEL_28;
            v49 = v68;
            if ( sub_1F58D20((__int64)&v63) )
            {
              v67.m128i_i8[0] = 0;
              v67.m128i_i64[1] = 0;
              LOBYTE(v58) = 0;
              sub_1F426C0(v46, v55, (unsigned int)v63, v49, (__int64)&v67, (unsigned int *)&v65, &v58);
              v37 = v58;
            }
            else
            {
              v35 = v46;
              sub_1F40D10((__int64)&v67, v46, v55, v63, v64);
              v36 = v67.m128i_u8[8];
              LOBYTE(v65) = v67.m128i_i8[8];
              v66 = v68;
              if ( v67.m128i_i8[8] )
              {
LABEL_28:
                v37 = *(_BYTE *)(v35 + v36 + 1155);
              }
              else
              {
                v50 = v68;
                if ( sub_1F58D20((__int64)&v65) )
                {
                  v67.m128i_i8[0] = 0;
                  v67.m128i_i64[1] = 0;
                  v57 = 0;
                  sub_1F426C0(v46, v55, (unsigned int)v65, v50, (__int64)&v67, &v58, &v57);
                  v37 = v57;
                }
                else
                {
                  sub_1F40D10((__int64)&v67, v46, v55, v65, v66);
                  v39 = v45;
                  LOBYTE(v39) = v67.m128i_i8[8];
                  v45 = v39;
                  v37 = sub_1D5E9F0(v46, v55, (unsigned int)v39, v68);
                }
              }
            }
          }
          v38 = sub_2045180(v37);
          v56 = (v38 + v54 - 1) / v38;
          goto LABEL_12;
        }
        v67.m128i_i8[0] = 0;
        v67.m128i_i64[1] = 0;
        LOBYTE(v63) = 0;
        v56 = sub_1F426C0(v46, v55, (unsigned int)v60, v61, (__int64)&v67, (unsigned int *)&v65, &v63);
      }
LABEL_12:
      v52 = *(_BYTE *)(v12[10] + v51);
      if ( (unsigned int)(*(_DWORD *)(*v10 + 504LL) - 34) <= 1 && *(_BYTE *)(v12[10] + v51) == 7 )
      {
        v56 = 1;
LABEL_15:
        v69 = v13;
        v27 = v10;
        v28 = v12;
        for ( i = 0; i != v56; ++i )
        {
          while ( 1 )
          {
            v31.m128i_i64[0] = (__int64)sub_1D2A660(
                                          v27,
                                          *(_DWORD *)(v28[13] + 4LL * (unsigned int)(i + v22)),
                                          v52,
                                          0,
                                          v17,
                                          v18);
            v30 = *(__m128i **)(v69 + 8);
            v67 = v31;
            if ( v30 != *(__m128i **)(v69 + 16) )
              break;
            ++i;
            sub_1D4B3A0((const __m128i **)v69, v30, &v67);
            if ( i == v56 )
              goto LABEL_21;
          }
          if ( v30 )
          {
            *v30 = v31;
            v30 = *(__m128i **)(v69 + 8);
          }
          *(_QWORD *)(v69 + 8) = v30 + 1;
        }
LABEL_21:
        v32 = i;
        v12 = v28;
        v10 = v27;
        v13 = v69;
        v22 += v32;
        goto LABEL_22;
      }
      if ( v56 )
        goto LABEL_15;
LABEL_22:
      v21.m128i_i64[0] = ++v51;
    }
    while ( v47 != v51 );
  }
  return v21.m128i_i64[0];
}
