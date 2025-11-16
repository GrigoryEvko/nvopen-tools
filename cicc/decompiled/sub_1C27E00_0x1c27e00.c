// Function: sub_1C27E00
// Address: 0x1c27e00
//
unsigned __int64 *__fastcall sub_1C27E00(unsigned __int64 *a1, __int64 a2, _QWORD *a3, char a4, char a5, char a6)
{
  _BYTE *v9; // rax
  _BYTE *v10; // rdx
  _BYTE *v11; // rsi
  _BYTE *v12; // rcx
  int v13; // edx
  __int64 v14; // r9
  _DWORD *v15; // r8
  __int64 v16; // rsi
  __int64 v17; // r9
  _DWORD *v18; // rax
  __int64 v19; // rcx
  unsigned int v20; // r15d
  int v21; // ebx
  int v22; // ebx
  int v23; // ebx
  char *v24; // r15
  unsigned int i; // esi
  __int64 v26; // rcx
  __int64 v27; // r15
  __int64 v28; // rbx
  int v29; // ebx
  __m128i *v30; // rbx
  __m128i *v31; // rax
  int v32; // r15d
  __int32 v33; // r15d
  __int64 v35; // r14
  __m128i *v36; // [rsp+8h] [rbp-78h]
  __m128i *v37; // [rsp+10h] [rbp-70h]
  int v40; // [rsp+1Ch] [rbp-64h]
  unsigned int v41; // [rsp+1Ch] [rbp-64h]
  __int64 v42; // [rsp+24h] [rbp-5Ch]
  int v43; // [rsp+2Ch] [rbp-54h]
  __m128i v44; // [rsp+30h] [rbp-50h] BYREF
  __int64 v45; // [rsp+40h] [rbp-40h]

  *a1 = 1;
  if ( *(_BYTE *)a2 )
  {
    v44.m128i_i64[0] = (__int64)"Endianness must be little endian [Supported: -e].";
    LOWORD(v45) = 259;
    sub_1C27C00(a1, v44.m128i_i64);
  }
  if ( !*(_DWORD *)(a2 + 12) )
  {
    if ( !*(_DWORD *)(a2 + 4) )
      goto LABEL_5;
LABEL_58:
    v44.m128i_i64[0] = (__int64)"Alloca address space must be 0 [Supported: -A:0].";
    LOWORD(v45) = 259;
    sub_1C27C00(a1, v44.m128i_i64);
    if ( a6 )
      goto LABEL_6;
    goto LABEL_15;
  }
  v44.m128i_i64[0] = (__int64)"Program address space must be 0 [Supported: -P:0].";
  LOWORD(v45) = 259;
  sub_1C27C00(a1, v44.m128i_i64);
  if ( *(_DWORD *)(a2 + 4) )
    goto LABEL_58;
LABEL_5:
  if ( a6 )
  {
LABEL_6:
    if ( *(_DWORD *)(a2 + 16) )
    {
      v44.m128i_i64[0] = (__int64)"No mangling mode supported.";
      LOWORD(v45) = 259;
      sub_1C27C00(a1, v44.m128i_i64);
    }
    v9 = *(_BYTE **)(a2 + 24);
    v10 = &v9[*(unsigned int *)(a2 + 32)];
    v11 = v9;
    v12 = *(_BYTE **)(a2 + 24);
    if ( v9 != v10 )
    {
      while ( *v12 != 16 )
      {
        if ( v10 == ++v12 )
          goto LABEL_14;
      }
      do
      {
        if ( *v11 == 64 )
          goto LABEL_18;
        ++v11;
      }
      while ( v10 != v11 );
    }
LABEL_14:
    v44.m128i_i64[0] = (__int64)"Require 16 and 64 bit native integer widths [Supported: -n:16:32:64].";
    LOWORD(v45) = 259;
    sub_1C27C00(a1, v44.m128i_i64);
  }
LABEL_15:
  v9 = *(_BYTE **)(a2 + 24);
  v10 = &v9[*(unsigned int *)(a2 + 32)];
  if ( v9 == v10 )
  {
LABEL_74:
    v44.m128i_i64[0] = (__int64)"Require 32 bit native integer width [Supported: -n:16:32:64 or -n:32].";
    LOWORD(v45) = 259;
    sub_1C27C00(a1, v44.m128i_i64);
  }
  else
  {
LABEL_18:
    while ( *v9 != 32 )
    {
      if ( v10 == ++v9 )
        goto LABEL_74;
    }
  }
  v13 = 0;
  v14 = 4LL * *(unsigned int *)(a2 + 416);
  v15 = (_DWORD *)(*(_QWORD *)(a2 + 408) + v14);
  v16 = v14 >> 4;
  v17 = v14 >> 2;
  while ( 1 )
  {
    if ( v16 )
    {
      v18 = *(_DWORD **)(a2 + 408);
      v19 = v16;
      while ( *v18 != v13 )
      {
        if ( v18[1] == v13 )
        {
          if ( v15 == v18 + 1 )
            goto LABEL_28;
          goto LABEL_41;
        }
        if ( v18[2] == v13 )
        {
          v18 += 2;
          goto LABEL_27;
        }
        if ( v18[3] == v13 )
        {
          v18 += 3;
          goto LABEL_27;
        }
        v18 += 4;
        if ( !--v19 )
        {
          v26 = v15 - v18;
          goto LABEL_45;
        }
      }
      goto LABEL_27;
    }
    v26 = v17;
    v18 = *(_DWORD **)(a2 + 408);
LABEL_45:
    if ( v26 == 2 )
      goto LABEL_77;
    if ( v26 == 3 )
    {
      if ( *v18 == v13 )
        goto LABEL_27;
      ++v18;
LABEL_77:
      if ( *v18 != v13 && *++v18 != v13 )
        goto LABEL_28;
      goto LABEL_27;
    }
    if ( v26 != 1 || *v18 != v13 )
      goto LABEL_28;
LABEL_27:
    if ( v15 != v18 )
      break;
LABEL_28:
    if ( ++v13 == 7 )
      goto LABEL_29;
  }
LABEL_41:
  v44.m128i_i64[0] = (__int64)"Non integral address spaces not supported.";
  LOWORD(v45) = 259;
  sub_1C27C00(a1, v44.m128i_i64);
LABEL_29:
  v20 = 0;
  while ( 1 )
  {
    v21 = sub_15A9480(a2, v20);
    if ( v21 != (unsigned int)sub_15A94D0(a2, v20) )
      break;
    v22 = sub_15A95A0(a2, v20);
    if ( v22 != (unsigned int)sub_15A9520(a2, v20) )
      break;
    if ( ++v20 == 7 )
    {
      v23 = sub_15A9480(a2, 0);
      if ( ((v23 - 4) & 0xFFFFFFFB) == 0 && v23 == (unsigned int)sub_15A9480(a2, 2u) )
      {
        v24 = (char *)&unk_42CC990;
        for ( i = 1; (unsigned int)sub_15A9480(a2, i) == 4 || v23 == (unsigned int)sub_15A9480(a2, i); i = *(_DWORD *)v24 )
        {
          v24 += 4;
          if ( "sampler" == v24 )
            goto LABEL_51;
        }
      }
      break;
    }
  }
  v44.m128i_i64[0] = (__int64)"Unsupported pointer alignment [Supported: 64-bit: -p:64:64:64, 32-bit (deprecated): -p:32:32:32].";
  LOWORD(v45) = 259;
  sub_1C27C00(a1, v44.m128i_i64);
LABEL_51:
  if ( (unsigned int)sub_15AAE10(a2, 1u) == 1 )
  {
    v29 = sub_15AAE10(a2, 1u);
    if ( v29 == (unsigned int)sub_15AAE30(a2, 1u) )
    {
      LODWORD(v45) = 16;
      v44.m128i_i64[0] = 0x200000001LL;
      v44.m128i_i64[1] = 0x800000004LL;
      v30 = (__m128i *)sub_22077B0(20);
      v36 = v30;
      *v30 = _mm_loadu_si128(&v44);
      v30[1].m128i_i32[0] = v45;
      v31 = (__m128i *)((char *)v30 + 20);
      if ( !a5 )
        v31 = v30 + 1;
      v37 = v31;
      if ( v30 == v31 )
      {
LABEL_85:
        j_j___libc_free_0(v36, 20);
        goto LABEL_53;
      }
      while ( 1 )
      {
        v33 = v30->m128i_i32[0];
        v41 = 8 * v30->m128i_i32[0];
        if ( v33 != (unsigned int)sub_15AAE10(a2, v41) )
          break;
        v32 = sub_15AAE10(a2, v41);
        if ( v32 != (unsigned int)sub_15AAE30(a2, v41) )
          break;
        v30 = (__m128i *)((char *)v30 + 4);
        if ( v37 == v30 )
          goto LABEL_85;
      }
      j_j___libc_free_0(v36, 20);
    }
  }
  v44.m128i_i64[0] = (__int64)"Unsupported integer alignment [Supported: -i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-i128:128:128].";
  LOWORD(v45) = 259;
  sub_1C27C00(a1, v44.m128i_i64);
LABEL_53:
  v43 = 16;
  v27 = 0;
  v42 = 0x800000004LL;
  v44.m128i_i64[0] = sub_16432A0(a3);
  v28 = v44.m128i_i64[0];
  v44.m128i_i64[1] = sub_16432B0(a3);
  v45 = sub_16432F0(a3);
  while ( (unsigned int)sub_15A9FE0(a2, v28) == *((_DWORD *)&v42 + v27) )
  {
    v40 = sub_15A9FE0(a2, v28);
    if ( v40 != (unsigned int)sub_15AAE50(a2, v28) )
      break;
    if ( ++v27 == 3 )
      goto LABEL_72;
    v28 = v44.m128i_i64[v27];
  }
  v44.m128i_i64[0] = (__int64)"Unsupported floating-point alignment [Supported: -f32:32:32-f64:64:64].";
  LOWORD(v45) = 259;
  sub_1C27C00(a1, v44.m128i_i64);
LABEL_72:
  if ( a4 )
  {
    v35 = sub_1645AC0(a3, 0);
    if ( (unsigned int)sub_15A9FE0(a2, v35) != 1 || (unsigned int)sub_15AAE50(a2, v35) != 1 )
    {
      v44.m128i_i64[0] = (__int64)"Unsupported aggregate alignment [Supported: -a:8:8].";
      LOWORD(v45) = 259;
      sub_1C27C00(a1, v44.m128i_i64);
    }
  }
  return a1;
}
