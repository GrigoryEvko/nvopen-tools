// Function: sub_3820160
// Address: 0x3820160
//
void __fastcall sub_3820160(__int64 *a1, unsigned __int64 a2, __int64 a3, __int64 a4, __m128i a5)
{
  __int64 v8; // rsi
  int v9; // ecx
  int v10; // eax
  __int64 v11; // rdx
  __m128i v12; // xmm2
  char v13; // r10
  unsigned __int16 v14; // dx
  __int64 v15; // r8
  _QWORD *v16; // r9
  __int32 v17; // edx
  int v18; // eax
  int v19; // ecx
  _WORD *v20; // rsi
  __int64 v21; // r10
  __int64 (__fastcall *v22)(__int64, __int64, unsigned int); // r9
  __int64 v23; // r8
  int v24; // eax
  __int64 v25; // [rsp+8h] [rbp-D8h]
  char v26; // [rsp+17h] [rbp-C9h]
  _QWORD *v27; // [rsp+18h] [rbp-C8h]
  __int64 v28; // [rsp+30h] [rbp-B0h] BYREF
  int v29; // [rsp+38h] [rbp-A8h]
  __m128i v30; // [rsp+40h] [rbp-A0h] BYREF
  __m128i v31; // [rsp+50h] [rbp-90h] BYREF
  __int64 v32[4]; // [rsp+60h] [rbp-80h] BYREF
  __int64 v33; // [rsp+80h] [rbp-60h] BYREF
  __int64 v34; // [rsp+88h] [rbp-58h]
  __int64 v35; // [rsp+90h] [rbp-50h]
  __int64 v36; // [rsp+98h] [rbp-48h]
  __int64 v37; // [rsp+A0h] [rbp-40h]

  v8 = *(_QWORD *)(a2 + 80);
  v28 = v8;
  if ( v8 )
    sub_B96E90((__int64)&v28, v8, 1);
  v9 = *(_DWORD *)(a2 + 72);
  v10 = *(_DWORD *)(a2 + 24);
  v11 = *(_QWORD *)(a2 + 40);
  v29 = v9;
  if ( v10 > 239 )
  {
    if ( (unsigned int)(v10 - 242) > 1 )
    {
LABEL_6:
      v12 = _mm_loadu_si128((const __m128i *)v11);
      v13 = 0;
      v31.m128i_i64[0] = 0;
      v31.m128i_i32[2] = 0;
      v30 = v12;
      goto LABEL_7;
    }
  }
  else if ( v10 <= 237 && (unsigned int)(v10 - 101) > 0x2F )
  {
    goto LABEL_6;
  }
  a5 = _mm_loadu_si128((const __m128i *)(v11 + 40));
  v13 = 1;
  v30 = a5;
  v31 = _mm_loadu_si128((const __m128i *)v11);
LABEL_7:
  v14 = *(_WORD *)(*(_QWORD *)(v30.m128i_i64[0] + 48) + 16LL * v30.m128i_u32[2]);
  v15 = v14;
  if ( v14 == 11 )
  {
    v16 = (_QWORD *)a1[1];
    LOWORD(v15) = 12;
    v33 = v28;
    if ( v28 )
    {
      v25 = v15;
      v26 = v13;
      v27 = v16;
      sub_B96E90((__int64)&v33, v28, 1);
      v9 = v29;
      v15 = v25;
      v13 = v26;
      v16 = v27;
    }
    LODWORD(v34) = v9;
    v30.m128i_i64[0] = (__int64)sub_38136F0(
                                  v30.m128i_i64[0],
                                  v30.m128i_i64[1],
                                  &v31,
                                  v13,
                                  v15,
                                  0,
                                  a5,
                                  (__int64)&v33,
                                  v16);
    v30.m128i_i32[2] = v17;
    if ( v33 )
      sub_B91220((__int64)&v33, v33);
    v18 = *(_DWORD *)(a2 + 24);
    if ( v18 != 275 && v18 != 135 )
    {
      if ( v18 != 137 && v18 != 277 )
      {
        if ( v18 != 276 && v18 != 136 )
        {
          if ( v18 == 278 || v18 == 138 )
            goto LABEL_20;
LABEL_70:
          BUG();
        }
        goto LABEL_62;
      }
LABEL_54:
      v19 = 302;
      goto LABEL_46;
    }
LABEL_45:
    v19 = 292;
    goto LABEL_46;
  }
  switch ( v10 )
  {
    case 135:
    case 275:
      if ( v14 == 12 )
        goto LABEL_45;
      v19 = 293;
      if ( v14 != 13 )
      {
        v19 = 294;
        if ( v14 != 14 )
        {
          v19 = 295;
          if ( v14 != 15 )
          {
            v19 = 296;
            if ( v14 != 16 )
              v19 = 729;
          }
        }
      }
      break;
    case 137:
    case 277:
      if ( v14 == 12 )
        goto LABEL_54;
      v19 = 303;
      if ( v14 != 13 )
      {
        v19 = 304;
        if ( v14 != 14 )
        {
          v19 = 305;
          if ( v14 != 15 )
          {
            v19 = 729;
            if ( v14 == 16 )
              v19 = 306;
          }
        }
      }
      break;
    case 136:
    case 276:
      if ( v14 == 12 )
      {
LABEL_62:
        v19 = 297;
        break;
      }
      v19 = 298;
      if ( v14 != 13 )
      {
        v19 = 299;
        if ( v14 != 14 )
        {
          v19 = 300;
          if ( v14 != 15 )
          {
            v19 = 729;
            if ( v14 == 16 )
              v19 = 301;
          }
        }
      }
      break;
    case 138:
    case 278:
      if ( v14 == 12 )
      {
LABEL_20:
        v19 = 307;
        break;
      }
      v19 = 308;
      if ( v14 != 13 )
      {
        v19 = 309;
        if ( v14 != 14 )
        {
          v19 = 310;
          if ( v14 != 15 )
          {
            v19 = 729;
            if ( v14 == 16 )
              v19 = 311;
          }
        }
      }
      break;
    default:
      goto LABEL_70;
  }
LABEL_46:
  v20 = (_WORD *)*a1;
  v21 = a1[1];
  v22 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int))(*(_QWORD *)(a2 + 48) + 8LL);
  v23 = **(unsigned __int16 **)(a2 + 48);
  v34 = 0;
  v35 = 0;
  v36 = 0;
  LOBYTE(v37) = 5;
  v33 = 0;
  sub_3494590(
    (__int64)v32,
    v20,
    v21,
    v19,
    v23,
    v22,
    (__int64)&v30,
    1u,
    0,
    0,
    0,
    0,
    5,
    (__int64)&v28,
    v31.m128i_i64[0],
    v31.m128i_i64[1]);
  sub_375BC20(a1, v32[0], v32[1], a3, a4, a5);
  v24 = *(_DWORD *)(a2 + 24);
  if ( v24 > 239 )
  {
    if ( (unsigned int)(v24 - 242) > 1 )
      goto LABEL_49;
  }
  else if ( v24 <= 237 && (unsigned int)(v24 - 101) > 0x2F )
  {
    goto LABEL_49;
  }
  sub_3760E70((__int64)a1, a2, 1, v32[2], v32[3]);
LABEL_49:
  if ( v28 )
    sub_B91220((__int64)&v28, v28);
}
