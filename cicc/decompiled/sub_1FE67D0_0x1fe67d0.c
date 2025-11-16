// Function: sub_1FE67D0
// Address: 0x1fe67d0
//
__int64 __fastcall sub_1FE67D0(
        __int64 a1,
        __int64 *a2,
        unsigned __int64 a3,
        int a4,
        unsigned int a5,
        __int64 a6,
        __int64 a7,
        char a8,
        unsigned __int8 a9,
        unsigned __int8 a10)
{
  bool v14; // bl
  __int64 v15; // r11
  bool v16; // al
  __int64 v17; // rdi
  char v18; // si
  __int64 i; // rdx
  _BYTE *v20; // rcx
  int v21; // ebx
  __int64 v22; // rsi
  __int64 v24; // rsi
  __int64 v25; // rax
  _QWORD *v26; // r14
  __int64 v27; // rax
  _QWORD *v28; // rax
  __int64 v29; // r8
  int v30; // r9d
  __int64 *v31; // r14
  __int64 v32; // rdi
  __int64 v33; // rax
  unsigned __int64 v34; // rdx
  __int64 v35; // [rsp+8h] [rbp-98h]
  __int64 v36; // [rsp+10h] [rbp-90h]
  __int64 v37; // [rsp+10h] [rbp-90h]
  __int32 v38; // [rsp+18h] [rbp-88h]
  __int64 v40; // [rsp+28h] [rbp-78h]
  __int64 v41; // [rsp+28h] [rbp-78h]
  __int64 v43; // [rsp+28h] [rbp-78h]
  int v44; // [rsp+3Ch] [rbp-64h]
  __m128i v45; // [rsp+40h] [rbp-60h] BYREF
  __int64 v46; // [rsp+50h] [rbp-50h]
  __int64 v47; // [rsp+58h] [rbp-48h]
  __int64 v48; // [rsp+60h] [rbp-40h]

  v14 = 0;
  v44 = sub_1FE6610((size_t *)a1, a3, a4, a7);
  v15 = *(_QWORD *)(a2[1] + 16);
  if ( a5 < *(unsigned __int16 *)(v15 + 2) )
    v14 = (*(_BYTE *)(*(_QWORD *)(v15 + 40) + 8LL * a5 + 2) & 4) != 0;
  if ( a6 )
  {
    if ( a5 < *(unsigned __int16 *)(a6 + 2) )
    {
      v24 = a6;
      v41 = v15;
      v25 = sub_1F3AD60(*(_QWORD *)(a1 + 16), v24, a5, *(_QWORD **)(a1 + 24), *(_QWORD *)a1);
      v15 = v41;
      v26 = (_QWORD *)v25;
      if ( v25 )
      {
        v36 = v41;
        v27 = sub_1E69410(*(__int64 **)(a1 + 8), v44, v25, 4u);
        v15 = v36;
        if ( !v27 )
        {
          v35 = v36;
          v28 = sub_1F4AAF0(*(_QWORD *)(a1 + 24), v26);
          v38 = sub_1E6B9A0(*(_QWORD *)(a1 + 8), (__int64)v28, (unsigned __int8 *)byte_3F871B3, 0, v29, v30);
          v31 = *(__int64 **)(a1 + 48);
          v37 = *(_QWORD *)(*(_QWORD *)(a1 + 40) + 56LL);
          v32 = *(_QWORD *)(a1 + 40);
          v43 = (__int64)sub_1E0B640(v37, *(_QWORD *)(*(_QWORD *)(a1 + 16) + 8LL) + 960LL, (__int64 *)(a3 + 72), 0);
          sub_1DD5BA0((__int64 *)(v32 + 16), v43);
          v33 = *(_QWORD *)v43;
          v34 = *v31 & 0xFFFFFFFFFFFFFFF8LL;
          *(_QWORD *)(v43 + 8) = v31;
          *(_QWORD *)v43 = v34 | v33 & 7;
          *(_QWORD *)(v34 + 8) = v43;
          *v31 = v43 | *v31 & 7;
          v45.m128i_i64[0] = 0x10000000;
          v46 = 0;
          v45.m128i_i32[2] = v38;
          v47 = 0;
          v48 = 0;
          sub_1E1A9C0(v43, v37, &v45);
          v45.m128i_i64[0] = 0;
          v46 = 0;
          v47 = 0;
          v45.m128i_i32[2] = v44;
          v48 = 0;
          sub_1E1A9C0(v43, v37, &v45);
          v44 = v38;
          v15 = v35;
        }
      }
    }
  }
  v40 = v15;
  v16 = sub_1D18C00(a3, 1, a4);
  if ( !v16 || *(_WORD *)(a3 + 24) == 47 )
  {
    v17 = a2[1];
    v21 = 2 * v14;
    if ( !a8 )
    {
LABEL_19:
      v14 = v21 != 0;
      v16 = 0;
      v18 = 0;
      goto LABEL_20;
    }
LABEL_26:
    v18 = 1;
    v14 = v21 != 0;
    v16 = 0;
    goto LABEL_20;
  }
  v17 = a2[1];
  if ( a8 )
  {
    if ( v14 )
    {
      v18 = v14;
      v16 = 0;
      goto LABEL_20;
    }
    v21 = 0;
    goto LABEL_26;
  }
  v18 = a9 | a10;
  if ( a9 | a10 )
  {
    if ( v14 )
    {
      v16 = 0;
      v18 = 0;
      goto LABEL_20;
    }
    v21 = 0;
    goto LABEL_19;
  }
  for ( i = *(unsigned int *)(v17 + 40); (_DWORD)i; i = (unsigned int)(i - 1) )
  {
    v20 = (_BYTE *)(*(_QWORD *)(v17 + 32) + 40LL * (unsigned int)(i - 1));
    if ( *v20 || (v20[3] & 0x20) == 0 )
      break;
  }
  if ( *(unsigned __int16 *)(v40 + 2) > (unsigned int)i )
  {
    if ( (*(_BYTE *)(*(_QWORD *)(v40 + 40) + 8 * i + 4) & 1) != 0 )
      v16 = 0;
    v18 = 0;
  }
LABEL_20:
  v45.m128i_i8[0] = 0;
  v46 = 0;
  v47 = 0;
  v48 = 0;
  v45.m128i_i8[3] = (v16 << 6) & 0x7F | (16 * v14) & 0x7F | v45.m128i_i8[3] & 0xF;
  v45.m128i_i16[1] &= 0xF00Fu;
  v45.m128i_i32[0] &= 0xFFF000FF;
  v45.m128i_i8[4] = v45.m128i_i8[4] & 0xF0 | (8 * (v18 & 1));
  v22 = *a2;
  v45.m128i_i32[2] = v44;
  return sub_1E1A9C0(v17, v22, &v45);
}
