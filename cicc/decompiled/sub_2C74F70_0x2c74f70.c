// Function: sub_2C74F70
// Address: 0x2c74f70
//
unsigned __int64 *__fastcall sub_2C74F70(unsigned __int64 *a1, __int64 a2, _QWORD *a3, char a4, char a5, char a6)
{
  void *v8; // rdx
  unsigned __int8 *v9; // r9
  __int64 v10; // rsi
  unsigned __int8 *v11; // rsi
  void *v12; // rdx
  unsigned __int8 *v13; // r9
  unsigned __int8 *v14; // r9
  void *v15; // rdx
  unsigned int v16; // ebx
  void *v17; // rdx
  unsigned int i; // r15d
  char v19; // bl
  void *v20; // rdx
  int v21; // ebx
  char v22; // bl
  void *v23; // rdx
  __int64 v24; // r15
  __int64 v25; // rbx
  unsigned __int64 v26; // rax
  char v27; // dl
  char v28; // al
  unsigned __int64 v29; // rdx
  char v31; // bl
  __m128i *v32; // rcx
  __m128i *v33; // rax
  __m128i *v34; // rbx
  char v35; // cl
  __int32 v36; // esi
  unsigned __int64 v37; // rax
  int v38; // r15d
  _QWORD *v39; // r15
  void *v40; // rdx
  unsigned int v41; // r15d
  unsigned int *v42; // rbx
  unsigned __int64 v43; // [rsp+0h] [rbp-90h]
  char v44; // [rsp+8h] [rbp-88h]
  unsigned int *v45; // [rsp+8h] [rbp-88h]
  unsigned __int8 v49; // [rsp+1Ch] [rbp-74h]
  char v50; // [rsp+1Ch] [rbp-74h]
  char v51; // [rsp+1Ch] [rbp-74h]
  __int64 v52; // [rsp+24h] [rbp-6Ch]
  int v53; // [rsp+2Ch] [rbp-64h]
  __m128i v54; // [rsp+30h] [rbp-60h] BYREF
  __int64 v55; // [rsp+40h] [rbp-50h]
  char v56; // [rsp+50h] [rbp-40h]
  char v57; // [rsp+51h] [rbp-3Fh]

  *a1 = 1;
  if ( *(_BYTE *)a2 )
  {
    v57 = 1;
    v54.m128i_i64[0] = (__int64)"Endianness must be little endian [Supported: -e].";
    v56 = 3;
    sub_2C74C40(a1, &v54, a3);
  }
  if ( *(_DWORD *)(a2 + 8) )
  {
    v57 = 1;
    v54.m128i_i64[0] = (__int64)"Program address space must be 0 [Supported: -P:0].";
    v56 = 3;
    sub_2C74C40(a1, &v54, a3);
    if ( !*(_DWORD *)(a2 + 12) )
    {
LABEL_5:
      v8 = (void *)*(unsigned int *)(a2 + 4);
      if ( !(_DWORD)v8 )
        goto LABEL_6;
      goto LABEL_34;
    }
  }
  else if ( !*(_DWORD *)(a2 + 12) )
  {
    goto LABEL_5;
  }
  v57 = 1;
  v54.m128i_i64[0] = (__int64)"Default global address space must be 0 [Supported: -G:0].";
  v56 = 3;
  sub_2C74C40(a1, &v54, a3);
  v8 = (void *)*(unsigned int *)(a2 + 4);
  if ( !(_DWORD)v8 )
  {
LABEL_6:
    if ( !a6 )
      goto LABEL_12;
    goto LABEL_7;
  }
LABEL_34:
  v57 = 1;
  v54.m128i_i64[0] = (__int64)"Alloca address space must be 0 [Supported: -A:0].";
  v56 = 3;
  sub_2C74C40(a1, &v54, v8);
  if ( !a6 )
    goto LABEL_12;
LABEL_7:
  if ( *(_DWORD *)(a2 + 24) )
  {
    v57 = 1;
    v54.m128i_i64[0] = (__int64)"No mangling mode supported.";
    v56 = 3;
    sub_2C74C40(a1, &v54, v8);
  }
  v9 = *(unsigned __int8 **)(a2 + 32);
  v10 = *(_QWORD *)(a2 + 40);
  v54.m128i_i64[0] = 16;
  v11 = &v9[v10];
  if ( v11 != sub_2C74700(v9, (__int64)v11, v54.m128i_i64) )
  {
    v54.m128i_i64[0] = 64;
    if ( v11 != sub_2C74700(v13, (__int64)v11, v54.m128i_i64) )
      goto LABEL_13;
  }
  v57 = 1;
  v54.m128i_i64[0] = (__int64)"Require 16 and 64 bit native integer widths [Supported: -n:16:32:64].";
  v56 = 3;
  sub_2C74C40(a1, &v54, v12);
LABEL_12:
  v14 = *(unsigned __int8 **)(a2 + 32);
  v11 = &v14[*(_QWORD *)(a2 + 40)];
LABEL_13:
  v54.m128i_i64[0] = 32;
  if ( v11 == sub_2C74700(v14, (__int64)v11, v54.m128i_i64) )
  {
    v57 = 1;
    v54.m128i_i64[0] = (__int64)"Require 32 bit native integer width [Supported: -n:16:32:64 or -n:32].";
    v56 = 3;
    sub_2C74C40(a1, &v54, v15);
  }
  v16 = 0;
  while ( !*((_BYTE *)sub_AE2980(a2, v16) + 16) )
  {
    if ( ++v16 == 7 )
      goto LABEL_18;
  }
  v57 = 1;
  v54.m128i_i64[0] = (__int64)"Non integral address spaces not supported.";
  v56 = 3;
  sub_2C74C40(a1, &v54, v17);
LABEL_18:
  for ( i = 0; i != 7; ++i )
  {
    v19 = sub_AE4370(a2, i);
    if ( (unsigned __int8)sub_AE4360(a2, i) != v19 )
      goto LABEL_24;
    v21 = sub_AE43D0(a2, i);
    if ( v21 != (unsigned int)sub_AE4380(a2, i) )
      goto LABEL_24;
  }
  v22 = sub_AE4360(a2, 0);
  v44 = v22;
  if ( (unsigned __int8)(v22 - 2) > 1u || v22 != (unsigned __int8)sub_AE4360(a2, 2u) )
    goto LABEL_24;
  v41 = 1;
  v42 = (unsigned int *)&unk_43A1980;
  if ( (unsigned __int8)sub_AE4360(a2, 1u) == 2 )
    goto LABEL_59;
LABEL_58:
  if ( v44 != (unsigned __int8)sub_AE4360(a2, v41) )
  {
LABEL_24:
    v57 = 1;
    v54.m128i_i64[0] = (__int64)"Unsupported pointer alignment [Supported: 64-bit: -p:64:64:64, 32-bit (deprecated): -p:32:32:32].";
    v56 = 3;
    sub_2C74C40(a1, &v54, v20);
    goto LABEL_25;
  }
LABEL_59:
  while ( &unk_43A1994 != (_UNKNOWN *)++v42 )
  {
    v41 = *v42;
    if ( (unsigned __int8)sub_AE4360(a2, *v42) != 2 )
      goto LABEL_58;
  }
LABEL_25:
  if ( !(unsigned __int8)sub_AE3FE0(a2, 1) )
  {
    v31 = sub_AE3FE0(a2, 1);
    if ( v31 == (unsigned __int8)sub_AE3FE0(a2, 1) )
    {
      LODWORD(v55) = 16;
      v54.m128i_i64[0] = 0x200000001LL;
      v54.m128i_i64[1] = 0x800000004LL;
      v32 = (__m128i *)sub_22077B0(0x14u);
      v43 = (unsigned __int64)v32;
      *v32 = _mm_loadu_si128(&v54);
      v32[1].m128i_i32[0] = v55;
      v33 = (__m128i *)((char *)v32 + 20);
      if ( !a5 )
        v33 = v32 + 1;
      v45 = (unsigned int *)v33;
      if ( v32 == v33 )
      {
LABEL_64:
        j_j___libc_free_0(v43);
        goto LABEL_27;
      }
      v34 = v32;
      while ( 1 )
      {
        v35 = -1;
        v36 = v34->m128i_i32[0];
        if ( v34->m128i_i32[0] )
        {
          _BitScanReverse64(&v37, v34->m128i_u32[0]);
          v35 = 63 - (v37 ^ 0x3F);
        }
        v38 = 8 * v36;
        if ( (unsigned __int8)sub_AE3FE0(a2, 8 * v36) != v35 )
          break;
        v51 = sub_AE3FE0(a2, v38);
        if ( v51 != (unsigned __int8)sub_AE3FE0(a2, v38) )
          break;
        v34 = (__m128i *)((char *)v34 + 4);
        if ( v45 == (unsigned int *)v34 )
          goto LABEL_64;
      }
      j_j___libc_free_0(v43);
    }
  }
  v57 = 1;
  v54.m128i_i64[0] = (__int64)"Unsupported integer alignment [Supported: -i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-i128:128:128].";
  v56 = 3;
  sub_2C74C40(a1, &v54, v23);
LABEL_27:
  v53 = 16;
  v52 = 0x800000004LL;
  v54.m128i_i64[0] = sub_BCB160(a3);
  v24 = v54.m128i_i64[0];
  v25 = 0;
  v54.m128i_i64[1] = sub_BCB170(a3);
  v55 = sub_BCB1B0(a3);
  v26 = 4;
LABEL_28:
  _BitScanReverse64(&v26, v26);
  v27 = 63 - (v26 ^ 0x3F);
  while ( 1 )
  {
    v49 = v27;
    v28 = sub_AE5020(a2, v24);
    v29 = v49;
    if ( v28 != v49 )
      break;
    v50 = sub_AE5260(a2, v24);
    if ( (unsigned __int8)sub_AE5020(a2, v24) != v50 )
      break;
    if ( ++v25 == 3 )
      goto LABEL_41;
    v26 = *(unsigned int *)((char *)&v52 + v25 * 4);
    v24 = v54.m128i_i64[v25];
    v27 = -1;
    if ( *(_DWORD *)((char *)&v52 + v25 * 4) )
      goto LABEL_28;
  }
  v57 = 1;
  v54.m128i_i64[0] = (__int64)"Unsupported floating-point alignment [Supported: -f32:32:32-f64:64:64].";
  v56 = 3;
  sub_2C74C40(a1, &v54, (void *)v29);
LABEL_41:
  if ( a4 )
  {
    v39 = sub_BD0E70(a3, 0);
    if ( (unsigned __int8)sub_AE5020(a2, (__int64)v39) || (unsigned __int8)sub_AE5260(a2, (__int64)v39) )
    {
      v57 = 1;
      v54.m128i_i64[0] = (__int64)"Unsupported aggregate alignment [Supported: -a:8:8].";
      v56 = 3;
      sub_2C74C40(a1, &v54, v40);
    }
  }
  return a1;
}
