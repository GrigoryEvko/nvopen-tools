// Function: sub_36DB420
// Address: 0x36db420
//
void __fastcall sub_36DB420(__int64 a1, __int64 a2, __m128i a3)
{
  __int64 v5; // rax
  _DWORD *v6; // rax
  unsigned __int64 *v7; // rsi
  int v8; // ebx
  __int64 v9; // rax
  int v10; // edx
  __int64 v11; // rcx
  __int64 v12; // rcx
  _QWORD *v13; // r13
  __int64 v14; // r10
  __int64 v15; // rax
  _QWORD *v16; // rsi
  unsigned __int8 *v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rcx
  __int64 v20; // r8
  __int64 v21; // rsi
  __int64 v22; // rdx
  _QWORD *v23; // rax
  int v24; // edx
  unsigned int v25; // r9d
  unsigned int v26; // eax
  int v27; // r9d
  int v28; // r10d
  __int64 v29; // r11
  int v30; // edi
  unsigned int v31; // eax
  int v32; // r9d
  __m128i v33; // xmm0
  unsigned __int64 *v34; // r9
  __m128i v35; // xmm2
  unsigned __int64 *v36; // rdx
  __m128i v37; // xmm0
  __int64 v38; // rax
  __m128i v39; // xmm0
  __m128i v40; // xmm3
  __m128i v41; // xmm0
  __int64 v42; // rbx
  unsigned __int64 *v43; // rdx
  __int64 v44; // rax
  const __m128i *v45; // rsi
  unsigned __int64 v46; // rcx
  unsigned __int64 v47; // rdx
  __m128i v48; // xmm0
  unsigned __int64 v49; // r8
  _QWORD *v50; // rdi
  __int64 v51; // r13
  __int64 v52; // rcx
  __int64 v53; // r8
  __int64 v54; // r9
  __m128i v55; // xmm0
  __m128i v56; // xmm1
  __m128i v57; // [rsp+0h] [rbp-F0h] BYREF
  __m128i v58; // [rsp+10h] [rbp-E0h] BYREF
  __int64 v59; // [rsp+20h] [rbp-D0h]
  unsigned __int64 *v60; // [rsp+28h] [rbp-C8h]
  unsigned __int64 *v61; // [rsp+30h] [rbp-C0h] BYREF
  int v62; // [rsp+38h] [rbp-B8h]
  __m128i v63; // [rsp+40h] [rbp-B0h] BYREF
  __m128i v64; // [rsp+50h] [rbp-A0h]
  __m128i *v65; // [rsp+60h] [rbp-90h] BYREF
  __int64 v66; // [rsp+68h] [rbp-88h]
  __m128i v67; // [rsp+70h] [rbp-80h] BYREF
  __m128i v68; // [rsp+80h] [rbp-70h]

  v5 = sub_2E79000(*(__int64 **)(*(_QWORD *)(a1 + 64) + 40LL));
  v6 = sub_AE2980(v5, 3u);
  v7 = *(unsigned __int64 **)(a2 + 80);
  v8 = v6[1];
  v61 = v7;
  if ( v7 )
  {
    sub_B96E90((__int64)&v61, (__int64)v7, 1);
    v7 = *(unsigned __int64 **)(a2 + 80);
  }
  v9 = *(_QWORD *)(a2 + 40);
  v10 = *(_DWORD *)(a2 + 72);
  v11 = *(_QWORD *)(v9 + 40);
  v62 = v10;
  v12 = *(_QWORD *)(v11 + 96);
  v13 = *(_QWORD **)(v12 + 24);
  if ( *(_DWORD *)(v12 + 32) > 0x40u )
    v13 = (_QWORD *)*v13;
  v65 = (__m128i *)v7;
  v14 = *(_QWORD *)(a1 + 64);
  if ( v7 )
  {
    v60 = *(unsigned __int64 **)(a1 + 64);
    sub_B96E90((__int64)&v65, (__int64)v7, 1);
    v9 = *(_QWORD *)(a2 + 40);
    v10 = *(_DWORD *)(a2 + 72);
    v14 = (__int64)v60;
  }
  LODWORD(v66) = v10;
  v15 = *(_QWORD *)(*(_QWORD *)(v9 + 80) + 96LL);
  v16 = *(_QWORD **)(v15 + 24);
  if ( *(_DWORD *)(v15 + 32) > 0x40u )
    v16 = (_QWORD *)*v16;
  v17 = sub_3400BD0(v14, (__int64)v16, (__int64)&v65, 7, 0, 1u, a3, 0);
  v19 = (__int64)v17;
  v20 = v18;
  if ( v65 )
  {
    v58.m128i_i64[0] = v18;
    v60 = (unsigned __int64 *)v17;
    sub_B91220((__int64)&v65, (__int64)v65);
    v20 = v58.m128i_i64[0];
    v19 = (__int64)v60;
  }
  v21 = *(_QWORD *)(a2 + 40);
  v22 = *(_QWORD *)(*(_QWORD *)(v21 + 80) + 96LL);
  v23 = *(_QWORD **)(v22 + 24);
  if ( *(_DWORD *)(v22 + 32) > 0x40u )
    v23 = (_QWORD *)*v23;
  v24 = *(_DWORD *)(a2 + 64);
  v25 = (unsigned int)v23;
  if ( (_DWORD)v13 != 9382 )
  {
    v26 = ((unsigned int)v23 >> 7) & 7;
    if ( v24 == 5 )
    {
      switch ( (v25 >> 15) & 0xF )
      {
        case 0u:
          if ( (_BYTE)v26 == 2 )
          {
            v28 = (v8 != 64) + 2962;
          }
          else if ( (unsigned __int8)v26 > 2u )
          {
            if ( (_BYTE)v26 != 3 )
              goto LABEL_315;
            v28 = (v8 == 64) + 2964;
          }
          else
          {
            v28 = (_BYTE)v26 == 0 ? 2954 : 2961;
          }
          goto LABEL_100;
        case 1u:
          if ( (_BYTE)v26 == 2 )
          {
            v28 = (v8 != 64) + 2974;
          }
          else if ( (unsigned __int8)v26 > 2u )
          {
            if ( (_BYTE)v26 != 3 )
              goto LABEL_315;
            v28 = (v8 == 64) + 2976;
          }
          else
          {
            v28 = (_BYTE)v26 == 0 ? 2966 : 2973;
          }
          goto LABEL_100;
        case 2u:
          if ( (_BYTE)v26 == 2 )
          {
            v28 = (v8 != 64) + 3082;
          }
          else if ( (unsigned __int8)v26 > 2u )
          {
            if ( (_BYTE)v26 != 3 )
              goto LABEL_315;
            v28 = (v8 == 64) + 3084;
          }
          else
          {
            v28 = (_BYTE)v26 == 0 ? 3074 : 3081;
          }
LABEL_100:
          v55 = _mm_loadu_si128((const __m128i *)(v21 + 120));
          v63.m128i_i64[0] = v19;
          v34 = (unsigned __int64 *)&v67;
          v63.m128i_i32[2] = v20;
          v56 = _mm_load_si128(&v63);
          v65 = &v67;
          v66 = 0x500000002LL;
          v64 = v55;
          v67 = v56;
          v68 = v55;
          goto LABEL_68;
        case 3u:
          if ( (_BYTE)v26 == 2 )
          {
            v28 = (v8 != 64) + 3094;
          }
          else if ( (unsigned __int8)v26 > 2u )
          {
            if ( (_BYTE)v26 != 3 )
              goto LABEL_315;
            v28 = (v8 == 64) + 3096;
          }
          else
          {
            v28 = (_BYTE)v26 == 0 ? 3086 : 3093;
          }
          goto LABEL_100;
        case 4u:
          if ( (_BYTE)v26 == 2 )
          {
            v28 = (v8 != 64) + 3058;
          }
          else if ( (unsigned __int8)v26 > 2u )
          {
            if ( (_BYTE)v26 != 3 )
              goto LABEL_315;
            v28 = (v8 == 64) + 3060;
          }
          else
          {
            v28 = (_BYTE)v26 == 0 ? 3050 : 3057;
          }
          goto LABEL_100;
        case 5u:
          if ( (_BYTE)v26 == 2 )
          {
            v28 = (v8 != 64) + 3070;
          }
          else if ( (unsigned __int8)v26 > 2u )
          {
            if ( (_BYTE)v26 != 3 )
              goto LABEL_315;
            v28 = (v8 == 64) + 3072;
          }
          else
          {
            v28 = (_BYTE)v26 == 0 ? 3062 : 3069;
          }
          goto LABEL_100;
        case 6u:
          if ( (_BYTE)v26 == 2 )
          {
            v28 = (v8 != 64) + 3034;
          }
          else if ( (unsigned __int8)v26 > 2u )
          {
            if ( (_BYTE)v26 != 3 )
              goto LABEL_315;
            v28 = (v8 == 64) + 3036;
          }
          else
          {
            v28 = (_BYTE)v26 == 0 ? 3026 : 3033;
          }
          goto LABEL_100;
        case 7u:
          if ( (_BYTE)v26 == 2 )
          {
            v28 = (v8 != 64) + 3046;
          }
          else if ( (unsigned __int8)v26 > 2u )
          {
            if ( (_BYTE)v26 != 3 )
              goto LABEL_315;
            v28 = (v8 == 64) + 3048;
          }
          else
          {
            v28 = (_BYTE)v26 == 0 ? 3038 : 3045;
          }
          goto LABEL_100;
        case 8u:
          if ( (_BYTE)v26 == 2 )
          {
            v28 = (v8 != 64) + 3022;
          }
          else if ( (unsigned __int8)v26 > 2u )
          {
            if ( (_BYTE)v26 != 3 )
              goto LABEL_315;
            v28 = (v8 == 64) + 3024;
          }
          else
          {
            v28 = (_BYTE)v26 == 0 ? 3002 : 3021;
          }
          goto LABEL_100;
        case 9u:
          if ( (_BYTE)v26 == 2 )
          {
            v28 = (v8 != 64) + 3011;
          }
          else if ( (unsigned __int8)v26 > 2u )
          {
            if ( (_BYTE)v26 != 3 )
              goto LABEL_315;
            v28 = (v8 == 64) + 3013;
          }
          else
          {
            v28 = (_BYTE)v26 == 0 ? 3003 : 3010;
          }
          goto LABEL_100;
        case 0xAu:
          if ( (_BYTE)v26 == 2 )
          {
            v28 = (v8 != 64) + 2998;
          }
          else if ( (unsigned __int8)v26 > 2u )
          {
            if ( (_BYTE)v26 != 3 )
              goto LABEL_315;
            v28 = (v8 == 64) + 3000;
          }
          else
          {
            v28 = (_BYTE)v26 == 0 ? 2978 : 2997;
          }
          goto LABEL_100;
        case 0xBu:
          if ( (_BYTE)v26 == 2 )
          {
            v28 = (v8 != 64) + 2987;
          }
          else if ( (unsigned __int8)v26 > 2u )
          {
            if ( (_BYTE)v26 != 3 )
              goto LABEL_315;
            v28 = (v8 == 64) + 2989;
          }
          else
          {
            v28 = (_BYTE)v26 == 0 ? 2979 : 2986;
          }
          goto LABEL_100;
        default:
          goto LABEL_316;
      }
    }
    v27 = v25 & 0x78000;
    switch ( v27 )
    {
      case 196608:
        if ( v24 == 6 )
        {
          if ( !(_BYTE)v26 )
          {
            v28 = 3114;
            goto LABEL_67;
          }
          if ( (_BYTE)v26 == 1 )
          {
            v28 = 3117;
            goto LABEL_67;
          }
          goto LABEL_315;
        }
        if ( v24 == 8 )
        {
          if ( !(_BYTE)v26 )
          {
            v28 = 3134;
            goto LABEL_67;
          }
          if ( (_BYTE)v26 == 1 )
          {
            v28 = 3137;
            goto LABEL_67;
          }
LABEL_315:
          sub_C64ED0("Invalid address space for nvvm.red", 1u);
        }
        break;
      case 294912:
        if ( v24 == 6 )
        {
          if ( !(_BYTE)v26 )
          {
            v28 = 3107;
            goto LABEL_67;
          }
          if ( (_BYTE)v26 == 1 )
          {
            v28 = 3110;
            goto LABEL_67;
          }
          goto LABEL_315;
        }
        if ( v24 == 8 )
        {
          if ( !(_BYTE)v26 )
          {
            v28 = 3127;
            goto LABEL_67;
          }
          if ( (_BYTE)v26 == 1 )
          {
            v28 = 3130;
            goto LABEL_67;
          }
          goto LABEL_315;
        }
        break;
      case 360448:
        if ( v24 == 6 )
        {
          if ( !(_BYTE)v26 )
          {
            v28 = 3099;
            goto LABEL_67;
          }
          if ( (_BYTE)v26 == 1 )
          {
            v28 = 3102;
            goto LABEL_67;
          }
          goto LABEL_315;
        }
        if ( v24 == 8 )
        {
          if ( !(_BYTE)v26 )
          {
            v28 = 3119;
            goto LABEL_67;
          }
          if ( (_BYTE)v26 == 1 )
          {
            v28 = 3122;
            goto LABEL_67;
          }
          goto LABEL_315;
        }
        break;
      case 262144:
        switch ( v24 )
        {
          case 6:
            if ( !(_BYTE)v26 )
            {
              v28 = 3106;
              goto LABEL_67;
            }
            if ( (_BYTE)v26 == 1 )
            {
              v28 = 3113;
              goto LABEL_67;
            }
            goto LABEL_315;
          case 8:
            if ( !(_BYTE)v26 )
            {
              v28 = 3126;
              goto LABEL_67;
            }
            if ( (_BYTE)v26 == 1 )
            {
              v28 = 3133;
              goto LABEL_67;
            }
            goto LABEL_315;
          case 12:
            if ( !(_BYTE)v26 )
            {
              v28 = 3142;
              goto LABEL_67;
            }
            if ( (_BYTE)v26 == 1 )
            {
              v28 = 3145;
              goto LABEL_67;
            }
            goto LABEL_315;
        }
        break;
      case 327680:
        switch ( v24 )
        {
          case 6:
            if ( !(_BYTE)v26 )
            {
              v28 = 3098;
              goto LABEL_67;
            }
            if ( (_BYTE)v26 == 1 )
            {
              v28 = 3105;
              goto LABEL_67;
            }
            goto LABEL_315;
          case 8:
            if ( !(_BYTE)v26 )
            {
              v28 = 3118;
              goto LABEL_67;
            }
            if ( (_BYTE)v26 == 1 )
            {
              v28 = 3125;
              goto LABEL_67;
            }
            goto LABEL_315;
          case 12:
            if ( !(_BYTE)v26 )
            {
              v28 = 3138;
LABEL_67:
              v39 = _mm_loadu_si128((const __m128i *)(v21 + 120));
              v63.m128i_i64[0] = v19;
              v34 = (unsigned __int64 *)&v67;
              v63.m128i_i32[2] = v20;
              v40 = _mm_load_si128(&v63);
              v65 = &v67;
              v66 = 0x500000002LL;
              v64 = v39;
              v67 = v40;
              v68 = v39;
LABEL_68:
              v30 = v24;
              goto LABEL_69;
            }
            if ( (_BYTE)v26 == 1 )
            {
              v28 = 3141;
              goto LABEL_67;
            }
            goto LABEL_315;
        }
        break;
    }
LABEL_41:
    sub_C64ED0("Invalid type and vector length for nvvm.red", 1u);
  }
  v29 = (unsigned int)(v24 - 1);
  v30 = v24 - 1;
  v31 = ((unsigned int)v23 >> 7) & 7;
  if ( v24 == 6 )
  {
    switch ( (v25 >> 15) & 0xF )
    {
      case 0u:
        if ( (_BYTE)v31 == 2 )
        {
          v28 = (v8 != 64) + 2957;
        }
        else if ( (unsigned __int8)v31 > 2u )
        {
          if ( (_BYTE)v31 != 3 )
            goto LABEL_315;
          v28 = (v8 == 64) + 2959;
        }
        else
        {
          v28 = 2955 - (((_BYTE)v31 == 0) - 1);
        }
        break;
      case 1u:
        if ( (_BYTE)v31 == 2 )
        {
          v28 = (v8 != 64) + 2969;
        }
        else if ( (unsigned __int8)v31 > 2u )
        {
          if ( (_BYTE)v31 != 3 )
            goto LABEL_315;
          v28 = (v8 == 64) + 2971;
        }
        else
        {
          v28 = 2967 - (((_BYTE)v31 == 0) - 1);
        }
        break;
      case 2u:
        if ( (_BYTE)v31 == 2 )
        {
          v28 = (v8 != 64) + 3077;
        }
        else if ( (unsigned __int8)v31 > 2u )
        {
          if ( (_BYTE)v31 != 3 )
            goto LABEL_315;
          v28 = (v8 == 64) + 3079;
        }
        else
        {
          v28 = 3075 - (((_BYTE)v31 == 0) - 1);
        }
        break;
      case 3u:
        if ( (_BYTE)v31 == 2 )
        {
          v28 = (v8 != 64) + 3089;
        }
        else if ( (unsigned __int8)v31 > 2u )
        {
          if ( (_BYTE)v31 != 3 )
            goto LABEL_315;
          v28 = (v8 == 64) + 3091;
        }
        else
        {
          v28 = 3087 - (((_BYTE)v31 == 0) - 1);
        }
        break;
      case 4u:
        if ( (_BYTE)v31 == 2 )
        {
          v28 = (v8 != 64) + 3053;
        }
        else if ( (unsigned __int8)v31 > 2u )
        {
          if ( (_BYTE)v31 != 3 )
            goto LABEL_315;
          v28 = (v8 == 64) + 3055;
        }
        else
        {
          v28 = 3051 - (((_BYTE)v31 == 0) - 1);
        }
        break;
      case 5u:
        if ( (_BYTE)v31 == 2 )
        {
          v28 = (v8 != 64) + 3065;
        }
        else if ( (unsigned __int8)v31 > 2u )
        {
          if ( (_BYTE)v31 != 3 )
            goto LABEL_315;
          v28 = (v8 == 64) + 3067;
        }
        else
        {
          v28 = 3063 - (((_BYTE)v31 == 0) - 1);
        }
        break;
      case 6u:
        if ( (_BYTE)v31 == 2 )
        {
          v28 = (v8 != 64) + 3029;
        }
        else if ( (unsigned __int8)v31 > 2u )
        {
          if ( (_BYTE)v31 != 3 )
            goto LABEL_315;
          v28 = (v8 == 64) + 3031;
        }
        else
        {
          v28 = 3027 - (((_BYTE)v31 == 0) - 1);
        }
        break;
      case 7u:
        if ( (_BYTE)v31 == 2 )
        {
          v28 = (v8 != 64) + 3041;
        }
        else if ( (unsigned __int8)v31 > 2u )
        {
          if ( (_BYTE)v31 != 3 )
            goto LABEL_315;
          v28 = (v8 == 64) + 3043;
        }
        else
        {
          v28 = 3039 - (((_BYTE)v31 == 0) - 1);
        }
        break;
      case 8u:
        if ( (_BYTE)v31 == 2 )
        {
          v28 = (v8 != 64) + 3017;
        }
        else if ( (unsigned __int8)v31 > 2u )
        {
          if ( (_BYTE)v31 != 3 )
            goto LABEL_315;
          v28 = (v8 == 64) + 3019;
        }
        else
        {
          v28 = 3015 - (((_BYTE)v31 == 0) - 1);
        }
        break;
      case 9u:
        if ( (_BYTE)v31 == 2 )
        {
          v28 = (v8 != 64) + 3006;
        }
        else if ( (unsigned __int8)v31 > 2u )
        {
          if ( (_BYTE)v31 != 3 )
            goto LABEL_315;
          v28 = (v8 == 64) + 3008;
        }
        else
        {
          v28 = 3004 - (((_BYTE)v31 == 0) - 1);
        }
        break;
      case 0xAu:
        if ( (_BYTE)v31 == 2 )
        {
          v28 = (v8 != 64) + 2993;
        }
        else if ( (unsigned __int8)v31 > 2u )
        {
          if ( (_BYTE)v31 != 3 )
            goto LABEL_315;
          v28 = (v8 == 64) + 2995;
        }
        else
        {
          v28 = 2991 - (((_BYTE)v31 == 0) - 1);
        }
        break;
      case 0xBu:
        if ( (_BYTE)v31 == 2 )
        {
          v28 = (v8 != 64) + 2982;
        }
        else if ( (unsigned __int8)v31 > 2u )
        {
          if ( (_BYTE)v31 != 3 )
            goto LABEL_315;
          v28 = (v8 == 64) + 2984;
        }
        else
        {
          v28 = 2980 - (((_BYTE)v31 == 0) - 1);
        }
        break;
      default:
LABEL_316:
        sub_C64ED0("Invalid Type for nvvm.red", 1u);
    }
  }
  else
  {
    v32 = v25 & 0x78000;
    switch ( v32 )
    {
      case 196608:
        if ( v24 == 7 )
        {
          if ( (_BYTE)v31 )
          {
            if ( (_BYTE)v31 != 1 )
              goto LABEL_315;
            v28 = 3116;
          }
          else
          {
            v28 = 3115;
          }
        }
        else
        {
          if ( v24 != 9 )
            goto LABEL_41;
          if ( (_BYTE)v31 )
          {
            if ( (_BYTE)v31 != 1 )
              goto LABEL_315;
            v28 = 3136;
          }
          else
          {
            v28 = 3135;
          }
        }
        break;
      case 294912:
        if ( v24 == 7 )
        {
          if ( (_BYTE)v31 )
          {
            if ( (_BYTE)v31 != 1 )
              goto LABEL_315;
            v28 = 3109;
          }
          else
          {
            v28 = 3108;
          }
        }
        else
        {
          if ( v24 != 9 )
            goto LABEL_41;
          if ( (_BYTE)v31 )
          {
            if ( (_BYTE)v31 != 1 )
              goto LABEL_315;
            v28 = 3129;
          }
          else
          {
            v28 = 3128;
          }
        }
        break;
      case 360448:
        if ( v24 == 7 )
        {
          if ( (_BYTE)v31 )
          {
            if ( (_BYTE)v31 != 1 )
              goto LABEL_315;
            v28 = 3101;
          }
          else
          {
            v28 = 3100;
          }
        }
        else
        {
          if ( v24 != 9 )
            goto LABEL_41;
          if ( (_BYTE)v31 )
          {
            if ( (_BYTE)v31 != 1 )
              goto LABEL_315;
            v28 = 3121;
          }
          else
          {
            v28 = 3120;
          }
        }
        break;
      case 262144:
        switch ( v24 )
        {
          case 7:
            if ( (_BYTE)v31 )
            {
              if ( (_BYTE)v31 != 1 )
                goto LABEL_315;
              v28 = 3112;
            }
            else
            {
              v28 = 3111;
            }
            break;
          case 9:
            if ( (_BYTE)v31 )
            {
              if ( (_BYTE)v31 != 1 )
                goto LABEL_315;
              v28 = 3132;
            }
            else
            {
              v28 = 3131;
            }
            break;
          case 13:
            if ( (_BYTE)v31 )
            {
              if ( (_BYTE)v31 != 1 )
                goto LABEL_315;
              v28 = 3144;
            }
            else
            {
              v28 = 3143;
            }
            break;
          default:
            goto LABEL_41;
        }
        break;
      case 327680:
        switch ( v24 )
        {
          case 7:
            if ( (_BYTE)v31 )
            {
              if ( (_BYTE)v31 != 1 )
                goto LABEL_315;
              v28 = 3104;
            }
            else
            {
              v28 = 3103;
            }
            break;
          case 9:
            if ( (_BYTE)v31 )
            {
              if ( (_BYTE)v31 != 1 )
                goto LABEL_315;
              v28 = 3124;
            }
            else
            {
              v28 = 3123;
            }
            break;
          case 13:
            if ( (_BYTE)v31 )
            {
              if ( (_BYTE)v31 != 1 )
                goto LABEL_315;
              v28 = 3140;
            }
            else
            {
              v28 = 3139;
            }
            break;
          default:
            goto LABEL_41;
        }
        break;
      default:
        goto LABEL_41;
    }
  }
  v33 = _mm_loadu_si128((const __m128i *)(v21 + 120));
  v63.m128i_i64[0] = v19;
  v34 = (unsigned __int64 *)&v67;
  v63.m128i_i32[2] = v20;
  v35 = _mm_load_si128(&v63);
  v65 = &v67;
  v66 = 0x500000002LL;
  v64 = v33;
  v67 = v35;
  v68 = v33;
  if ( (unsigned int)v29 <= 4 )
  {
    v36 = (unsigned __int64 *)&v67;
    v37 = _mm_loadu_si128((const __m128i *)(v21 + 40 * v29));
    v38 = 4;
LABEL_84:
    *(__m128i *)&v36[v38] = v37;
    v45 = *(const __m128i **)(a2 + 40);
    v46 = HIDWORD(v66);
    LODWORD(v66) = v66 + 1;
    v44 = (unsigned int)v66;
    v47 = (unsigned int)v66 + 1LL;
    goto LABEL_75;
  }
LABEL_69:
  v41 = _mm_loadu_si128((const __m128i *)(v21 + 160));
  v42 = 200;
  v43 = (unsigned __int64 *)&v67;
  v20 = 40LL * (unsigned int)(v30 - 5) + 200;
  v44 = 2;
  while ( 1 )
  {
    *(__m128i *)&v43[2 * v44] = v41;
    v44 = (unsigned int)(v66 + 1);
    LODWORD(v66) = v66 + 1;
    if ( v20 == v42 )
      break;
    v41 = _mm_loadu_si128((const __m128i *)(*(_QWORD *)(a2 + 40) + v42));
    if ( v44 + 1 > (unsigned __int64)HIDWORD(v66) )
    {
      v59 = v20;
      v57.m128i_i32[0] = v28;
      v60 = v34;
      v58 = v41;
      sub_C8D5F0((__int64)&v65, v34, v44 + 1, 0x10u, v20, (__int64)v34);
      v44 = (unsigned int)v66;
      v20 = v59;
      v28 = v57.m128i_i32[0];
      v41 = _mm_load_si128(&v58);
      v34 = v60;
    }
    v43 = (unsigned __int64 *)v65;
    v42 += 40;
  }
  v45 = *(const __m128i **)(a2 + 40);
  v46 = HIDWORD(v66);
  v47 = v44 + 1;
  if ( (_DWORD)v13 == 9382 )
  {
    v37 = _mm_loadu_si128((const __m128i *)((char *)v45 + 40 * (unsigned int)(*(_DWORD *)(a2 + 64) - 1)));
    if ( v47 > HIDWORD(v66) )
    {
      v58.m128i_i32[0] = v28;
      v60 = v34;
      v57 = v37;
      sub_C8D5F0((__int64)&v65, v34, v47, 0x10u, v20, (__int64)v34);
      v36 = (unsigned __int64 *)v65;
      v37 = _mm_load_si128(&v57);
      v28 = v58.m128i_i32[0];
      v34 = v60;
      v38 = 2LL * (unsigned int)v66;
    }
    else
    {
      v36 = (unsigned __int64 *)v65;
      v38 = 2 * v44;
    }
    goto LABEL_84;
  }
LABEL_75:
  v48 = _mm_loadu_si128(v45);
  if ( v46 < v47 )
  {
    v57.m128i_i32[0] = v28;
    v60 = v34;
    v58 = v48;
    sub_C8D5F0((__int64)&v65, v34, v47, 0x10u, v20, (__int64)v34);
    v44 = (unsigned int)v66;
    v28 = v57.m128i_i32[0];
    v48 = _mm_load_si128(&v58);
    v34 = v60;
  }
  v65[v44] = v48;
  v49 = *(_QWORD *)(a2 + 48);
  v50 = *(_QWORD **)(a1 + 64);
  v60 = v34;
  LODWORD(v66) = v66 + 1;
  v51 = sub_33E66D0(
          v50,
          v28,
          (__int64)&v61,
          v49,
          *(unsigned int *)(a2 + 68),
          (__int64)v34,
          (unsigned __int64 *)v65,
          (unsigned int)v66);
  sub_34158F0(*(_QWORD *)(a1 + 64), a2, v51, v52, v53, v54);
  sub_3421DB0(v51);
  sub_33ECEA0(*(const __m128i **)(a1 + 64), a2);
  if ( v65 != (__m128i *)v60 )
    _libc_free((unsigned __int64)v65);
  if ( v61 )
    sub_B91220((__int64)&v61, (__int64)v61);
}
