// Function: sub_24A8B90
// Address: 0x24a8b90
//
__int64 __fastcall sub_24A8B90(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v6; // rdx
  char *v7; // rax
  __int64 v8; // rdx
  size_t v9; // rdx
  __m128i *v10; // rsi
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rax
  __int64 v14; // rcx
  unsigned int v15; // esi
  __int64 *v16; // rdx
  __int64 v17; // r8
  __int64 v18; // r14
  __int64 *v19; // rdx
  __int64 v20; // r14
  __int64 v21; // rbx
  __m128i *v22; // rdx
  __m128i *v23; // rdx
  __m128i si128; // xmm0
  __int64 v26; // rax
  __int64 v27; // rdx
  __int64 v28; // rdi
  __int64 v29; // rax
  __int64 v30; // rax
  int v31; // edx
  int v32; // r9d
  __m128i *v33; // [rsp+10h] [rbp-F0h] BYREF
  size_t v34; // [rsp+18h] [rbp-E8h]
  __m128i v35; // [rsp+20h] [rbp-E0h] BYREF
  __m128i *v36; // [rsp+30h] [rbp-D0h] BYREF
  size_t v37; // [rsp+38h] [rbp-C8h]
  __m128i v38; // [rsp+40h] [rbp-C0h] BYREF
  _QWORD v39[3]; // [rsp+50h] [rbp-B0h] BYREF
  __int64 v40; // [rsp+68h] [rbp-98h]
  __m128i *v41; // [rsp+70h] [rbp-90h]
  __int64 v42; // [rsp+78h] [rbp-88h]
  __int64 v43; // [rsp+80h] [rbp-80h]
  unsigned __int64 v44[14]; // [rsp+90h] [rbp-70h] BYREF

  *(_QWORD *)a1 = a1 + 16;
  *(_QWORD *)(a1 + 8) = 0;
  *(_BYTE *)(a1 + 16) = 0;
  v43 = a1;
  v39[1] = 0;
  v39[2] = 0;
  v40 = 0;
  v41 = 0;
  v42 = 0x100000000LL;
  v39[0] = &unk_49DD210;
  sub_CB5980((__int64)v39, 0, 0, 0);
  sub_BD5D20(a2);
  if ( v6 )
  {
    v7 = (char *)sub_BD5D20(a2);
    if ( v7 )
    {
      v33 = &v35;
      sub_24A2F70((__int64 *)&v33, v7, (__int64)&v7[v8]);
      v9 = v34;
      v10 = v33;
    }
    else
    {
      v35.m128i_i8[0] = 0;
      v9 = 0;
      v33 = &v35;
      v10 = &v35;
      v34 = 0;
    }
  }
  else
  {
    v44[0] = (unsigned __int64)&unk_49DD210;
    v44[6] = (unsigned __int64)&v36;
    v44[5] = 0x100000000LL;
    v36 = &v38;
    v37 = 0;
    v38.m128i_i8[0] = 0;
    memset(&v44[1], 0, 32);
    sub_CB5980((__int64)v44, 0, 0, 0);
    sub_A5BF40((unsigned __int8 *)a2, (__int64)v44, 0, 0);
    v33 = &v35;
    if ( v36 == &v38 )
    {
      v35 = _mm_load_si128(&v38);
    }
    else
    {
      v33 = v36;
      v35.m128i_i64[0] = v38.m128i_i64[0];
    }
    v36 = &v38;
    v34 = v37;
    v38.m128i_i8[0] = 0;
    v37 = 0;
    v44[0] = (unsigned __int64)&unk_49DD210;
    sub_CB5840((__int64)v44);
    if ( v36 != &v38 )
      j_j___libc_free_0((unsigned __int64)v36);
    v9 = v34;
    v10 = v33;
  }
  v11 = sub_CB6200((__int64)v39, (unsigned __int8 *)v10, v9);
  v12 = *(_QWORD *)(v11 + 32);
  if ( (unsigned __int64)(*(_QWORD *)(v11 + 24) - v12) <= 2 )
  {
    sub_CB6200(v11, ":\\l", 3u);
  }
  else
  {
    *(_BYTE *)(v12 + 2) = 108;
    *(_WORD *)v12 = 23610;
    *(_QWORD *)(v11 + 32) += 3LL;
  }
  if ( v33 != &v35 )
    j_j___libc_free_0((unsigned __int64)v33);
  v13 = *(unsigned int *)(a3 + 296);
  v14 = *(_QWORD *)(a3 + 280);
  if ( (_DWORD)v13 )
  {
    v15 = (v13 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v16 = (__int64 *)(v14 + 16LL * v15);
    v17 = *v16;
    if ( a2 == *v16 )
    {
LABEL_15:
      if ( v16 != (__int64 *)(v14 + 16 * v13) )
      {
        v18 = v16[1];
        goto LABEL_17;
      }
    }
    else
    {
      v31 = 1;
      while ( v17 != -4096 )
      {
        v32 = v31 + 1;
        v15 = (v13 - 1) & (v31 + v15);
        v16 = (__int64 *)(v14 + 16LL * v15);
        v17 = *v16;
        if ( a2 == *v16 )
          goto LABEL_15;
        v31 = v32;
      }
    }
  }
  v18 = 0;
LABEL_17:
  if ( (unsigned __int64)(v40 - (_QWORD)v41) <= 7 )
  {
    sub_CB6200((__int64)v39, "Count : ", 8u);
  }
  else
  {
    v41->m128i_i64[0] = 0x203A20746E756F43LL;
    v41 = (__m128i *)((char *)v41 + 8);
  }
  if ( v18 && *(_BYTE *)(v18 + 24) )
  {
    v30 = sub_CB59D0((__int64)v39, *(_QWORD *)(v18 + 16));
    sub_904010(v30, "\\l");
  }
  else
  {
    v19 = (__int64 *)v41;
    if ( (unsigned __int64)(v40 - (_QWORD)v41) <= 8 )
    {
      sub_CB6200((__int64)v39, "Unknown\\l", 9u);
    }
    else
    {
      v41->m128i_i8[8] = 108;
      *v19 = 0x5C6E776F6E6B6E55LL;
      v41 = (__m128i *)((char *)v41 + 9);
    }
  }
  if ( (_BYTE)qword_4FEBC88 )
  {
    v20 = *(_QWORD *)(a2 + 56);
    v21 = a2 + 48;
    if ( v20 != v21 )
    {
      while ( 1 )
      {
        if ( !v20 )
          BUG();
        if ( *(_BYTE *)(v20 - 24) != 86 )
          goto LABEL_26;
        v22 = v41;
        if ( (unsigned __int64)(v40 - (_QWORD)v41) <= 0xE )
        {
          sub_CB6200((__int64)v39, "SELECT : { T = ", 0xFu);
        }
        else
        {
          v41->m128i_i32[2] = 1411414816;
          v22->m128i_i64[0] = 0x3A205443454C4553LL;
          v22->m128i_i16[6] = 15648;
          v22->m128i_i8[14] = 32;
          v41 = (__m128i *)((char *)v41 + 15);
        }
        if ( (unsigned __int8)sub_BC8C50(v20 - 24, &v36, v44) )
          break;
        v23 = v41;
        if ( (unsigned __int64)(v40 - (_QWORD)v41) <= 0x17 )
        {
          sub_CB6200((__int64)v39, "Unknown, F = Unknown }\\l", 0x18u);
LABEL_26:
          v20 = *(_QWORD *)(v20 + 8);
          if ( v21 == v20 )
            goto LABEL_34;
        }
        else
        {
          si128 = _mm_load_si128((const __m128i *)&xmmword_42B6820);
          v41[1].m128i_i64[0] = 0x6C5C7D206E776F6ELL;
          *v23 = si128;
          v41 = (__m128i *)((char *)v41 + 24);
          v20 = *(_QWORD *)(v20 + 8);
          if ( v21 == v20 )
            goto LABEL_34;
        }
      }
      v26 = sub_CB59D0((__int64)v39, (unsigned __int64)v36);
      v27 = *(_QWORD *)(v26 + 32);
      v28 = v26;
      if ( (unsigned __int64)(*(_QWORD *)(v26 + 24) - v27) <= 5 )
      {
        v28 = sub_CB6200(v26, ", F = ", 6u);
      }
      else
      {
        *(_DWORD *)v27 = 541466668;
        *(_WORD *)(v27 + 4) = 8253;
        *(_QWORD *)(v26 + 32) += 6LL;
      }
      v29 = sub_CB59D0(v28, v44[0]);
      sub_904010(v29, " }\\l");
      goto LABEL_26;
    }
  }
LABEL_34:
  v39[0] = &unk_49DD210;
  sub_CB5840((__int64)v39);
  return a1;
}
