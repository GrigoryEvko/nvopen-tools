// Function: sub_36DCF60
// Address: 0x36dcf60
//
__int64 __fastcall sub_36DCF60(int a1, int a2, __int64 a3)
{
  unsigned int v4; // edx
  __int64 v5; // rax
  __m128i si128; // xmm0
  __int64 result; // rax
  __int64 v8; // r9
  __int64 v9; // rdx
  const char *v10; // rdx
  __m128i *v11; // [rsp+0h] [rbp-180h] BYREF
  __int64 v12; // [rsp+8h] [rbp-178h]
  __m128i v13; // [rsp+10h] [rbp-170h] BYREF
  __m128i *v14; // [rsp+20h] [rbp-160h] BYREF
  __int64 v15; // [rsp+28h] [rbp-158h]
  __m128i v16; // [rsp+30h] [rbp-150h] BYREF
  __m128i v17; // [rsp+40h] [rbp-140h] BYREF
  __int64 v18; // [rsp+50h] [rbp-130h]
  __m128i v19; // [rsp+58h] [rbp-128h] BYREF
  __int64 v20; // [rsp+70h] [rbp-110h] BYREF
  __m128i *v21; // [rsp+78h] [rbp-108h]
  __int64 v22; // [rsp+80h] [rbp-100h]
  __m128i v23; // [rsp+88h] [rbp-F8h] BYREF
  void *v24; // [rsp+98h] [rbp-E8h]
  __m128i *v25; // [rsp+A0h] [rbp-E0h]
  __int64 v26; // [rsp+A8h] [rbp-D8h]
  __m128i v27; // [rsp+B0h] [rbp-D0h] BYREF
  __m128i v28; // [rsp+C0h] [rbp-C0h] BYREF
  _QWORD *v29; // [rsp+D0h] [rbp-B0h] BYREF
  __m128i v30; // [rsp+D8h] [rbp-A8h] BYREF
  void *v31; // [rsp+E8h] [rbp-98h] BYREF
  __m128i *v32; // [rsp+F0h] [rbp-90h]
  __int64 v33; // [rsp+F8h] [rbp-88h]
  __m128i v34; // [rsp+100h] [rbp-80h] BYREF
  void *v35; // [rsp+110h] [rbp-70h] BYREF
  __m128i *v36; // [rsp+118h] [rbp-68h]
  __int64 v37; // [rsp+120h] [rbp-60h]
  __m128i v38; // [rsp+128h] [rbp-58h] BYREF
  _QWORD v39[9]; // [rsp+138h] [rbp-48h] BYREF

  if ( a2 == 2 )
  {
    v20 = 20;
    v28.m128i_i64[0] = (__int64)&v29;
    v5 = sub_22409D0((__int64)&v28, (unsigned __int64 *)&v20, 0);
    si128 = _mm_load_si128((const __m128i *)&xmmword_451CEB0);
    v28.m128i_i64[0] = v5;
    v29 = (_QWORD *)v20;
    *(_DWORD *)(v5 + 16) = 1701015141;
    *(__m128i *)v5 = si128;
    v28.m128i_i64[1] = v20;
    *(_BYTE *)(v28.m128i_i64[0] + v20) = 0;
    sub_305B5A0(a3, (__int64)&v28);
    if ( (_QWORD **)v28.m128i_i64[0] != &v29 )
      j_j___libc_free_0(v28.m128i_u64[0]);
  }
  v4 = *(_DWORD *)(a3 + 344);
  if ( (v4 <= 0x59 || *(_DWORD *)(a3 + 336) <= 0x55u) && (unsigned int)(a1 - 4) <= 1 )
  {
LABEL_6:
    switch ( a2 )
    {
      case 0:
        goto LABEL_33;
      case 1:
        result = 1668;
        if ( v4 > 0x45 )
          return *(_DWORD *)(a3 + 336) < 0x3Cu ? 1668 : 6973;
        return result;
      case 2:
        return 6972;
      case 3:
        result = 1669;
        if ( v4 > 0x45 )
          return *(_DWORD *)(a3 + 336) < 0x3Cu ? 1669 : 6974;
        return result;
      case 4:
        result = 1670;
        if ( v4 > 0x45 )
          return *(_DWORD *)(a3 + 336) < 0x3Cu ? 1670 : 6975;
        return result;
      default:
        goto LABEL_63;
    }
  }
  switch ( a1 )
  {
    case 0:
    case 2:
    case 8:
    case 9:
      sub_35EE190((__int64)&v14, a2);
      sub_35EDF20((__int64)&v11, a1);
      v28.m128i_i64[1] = (__int64)&v30;
      if ( v14 == &v16 )
      {
        v30 = _mm_load_si128(&v16);
      }
      else
      {
        v28.m128i_i64[1] = (__int64)v14;
        v30.m128i_i64[0] = v16.m128i_i64[0];
      }
      v14 = &v16;
      v8 = v15;
      v17.m128i_i64[0] = (__int64)&unk_49E64B0;
      v15 = 0;
      v16.m128i_i8[0] = 0;
      v17.m128i_i64[1] = (__int64)&v19;
      if ( v11 == &v13 )
      {
        v19 = _mm_load_si128(&v13);
      }
      else
      {
        v17.m128i_i64[1] = (__int64)v11;
        v19.m128i_i64[0] = v13.m128i_i64[0];
      }
      v9 = v12;
      v11 = &v13;
      v12 = 0;
      v18 = v9;
      v13.m128i_i8[0] = 0;
      v21 = &v23;
      if ( (__m128i *)v28.m128i_i64[1] == &v30 )
      {
        v23 = _mm_loadu_si128(&v30);
      }
      else
      {
        v21 = (__m128i *)v28.m128i_i64[1];
        v23.m128i_i64[0] = v30.m128i_i64[0];
      }
      v24 = &unk_49E64B0;
      v25 = &v27;
      if ( (__m128i *)v17.m128i_i64[1] == &v19 )
      {
        v27 = _mm_loadu_si128(&v19);
      }
      else
      {
        v25 = (__m128i *)v17.m128i_i64[1];
        v27.m128i_i64[0] = v19.m128i_i64[0];
      }
      v26 = v9;
      v29 = v39;
      v31 = &unk_49E64B0;
      v32 = &v34;
      v28.m128i_i64[0] = (__int64)"Unsupported \"{}\" ordering and \"{}\" scope for fence.";
      v28.m128i_i64[1] = 51;
      v30.m128i_i64[0] = 2;
      v30.m128i_i8[8] = 1;
      if ( v21 == &v23 )
      {
        v34 = _mm_loadu_si128(&v23);
      }
      else
      {
        v32 = v21;
        v34.m128i_i64[0] = v23.m128i_i64[0];
      }
      v33 = v8;
      v35 = &unk_49E64B0;
      v36 = &v38;
      v21 = &v23;
      v22 = 0;
      v23.m128i_i8[0] = 0;
      if ( v25 == &v27 )
      {
        v38 = _mm_load_si128(&v27);
      }
      else
      {
        v36 = v25;
        v38.m128i_i64[0] = v27.m128i_i64[0];
      }
      v37 = v9;
      v39[0] = &v35;
      v39[1] = &v31;
      v23.m128i_i16[4] = 263;
      v20 = (__int64)&v28;
      sub_C64D30((__int64)&v20, 1u);
    case 4:
      switch ( a2 )
      {
        case 0:
          goto LABEL_33;
        case 1:
          result = 1668;
          if ( v4 > 0x45 )
            return *(_DWORD *)(a3 + 336) < 0x3Cu ? 1668 : 6977;
          return result;
        case 2:
          return 6976;
        case 3:
          result = 1669;
          if ( v4 > 0x45 )
            return *(_DWORD *)(a3 + 336) < 0x3Cu ? 1669 : 6978;
          return result;
        case 4:
          result = 1670;
          if ( v4 > 0x45 )
            return *(_DWORD *)(a3 + 336) < 0x3Cu ? 1670 : 6979;
          return result;
        default:
          goto LABEL_63;
      }
    case 5:
      switch ( a2 )
      {
        case 0:
LABEL_33:
          sub_35EE190((__int64)&v17, 0);
          v10 = "Unsupported scope \"{}\" for acquire/release/acq_rel fence.";
          goto LABEL_34;
        case 1:
          result = 1668;
          if ( v4 > 0x45 )
            return *(_DWORD *)(a3 + 336) < 0x3Cu ? 1668 : 6981;
          return result;
        case 2:
          return 6980;
        case 3:
          result = 1669;
          if ( v4 > 0x45 )
            return *(_DWORD *)(a3 + 336) < 0x3Cu ? 1669 : 6982;
          return result;
        case 4:
          result = 1670;
          if ( v4 > 0x45 )
            return *(_DWORD *)(a3 + 336) < 0x3Cu ? 1670 : 6983;
          return result;
        default:
          goto LABEL_63;
      }
    case 6:
      goto LABEL_6;
    case 7:
      switch ( a2 )
      {
        case 0:
          sub_35EE190((__int64)&v17, 0);
          v10 = "Unsupported scope \"{}\" for seq_cst fence.";
LABEL_34:
          sub_35EF270(&v28, 1, v10, &v17);
          v23.m128i_i16[4] = 263;
          v20 = (__int64)&v28;
          sub_C64D30((__int64)&v20, 1u);
        case 1:
          result = 1668;
          if ( v4 > 0x45 )
            return *(_DWORD *)(a3 + 336) < 0x3Cu ? 1668 : 6985;
          return result;
        case 2:
          return 6984;
        case 3:
          result = 1669;
          if ( v4 > 0x45 )
            return *(_DWORD *)(a3 + 336) < 0x3Cu ? 1669 : 6986;
          return result;
        case 4:
          result = 1670;
          if ( v4 > 0x45 )
            return *(_DWORD *)(a3 + 336) < 0x3Cu ? 1670 : 6987;
          return result;
        default:
          goto LABEL_63;
      }
      goto LABEL_63;
    default:
LABEL_63:
      BUG();
  }
}
