// Function: sub_D13130
// Address: 0xd13130
//
__int64 __fastcall sub_D13130(__int64 a1, __int64 *a2, __int64 a3, __int64 a4)
{
  __int64 v5; // rax
  __int64 v6; // rdi
  __int64 v7; // rbx
  __m128i *v8; // rdx
  __m128i si128; // xmm0
  __int64 v10; // rsi
  __int64 v11; // rdi
  __int64 v12; // rdx
  __int64 v13; // rax
  _WORD *v14; // rdx
  char *v15; // r13
  __int64 v16; // r15
  __int64 *v17; // rbx
  __int64 v18; // rdi
  const char *v19; // rax
  size_t v20; // rdx
  void *v21; // rdi
  unsigned __int8 *v22; // rsi
  __int64 v23; // rdi
  __int64 v24; // rcx
  _WORD *v25; // rdx
  __int64 v26; // rdi
  _BYTE *v27; // rax
  __int64 v29; // rdi
  __m128i *v30; // rdx
  __m128i v31; // xmm0
  int v33; // [rsp+14h] [rbp-ACh]
  size_t v34; // [rsp+18h] [rbp-A8h]
  __int64 v35; // [rsp+18h] [rbp-A8h]
  _QWORD v36[2]; // [rsp+20h] [rbp-A0h] BYREF
  __int64 v37; // [rsp+30h] [rbp-90h]
  __int64 v38; // [rsp+38h] [rbp-88h]
  __int64 v39; // [rsp+40h] [rbp-80h]
  __int64 v40; // [rsp+48h] [rbp-78h]
  __int64 v41; // [rsp+50h] [rbp-70h]
  __int64 v42; // [rsp+58h] [rbp-68h]
  char *v43; // [rsp+60h] [rbp-60h]
  char *v44; // [rsp+68h] [rbp-58h]
  __int64 v45; // [rsp+70h] [rbp-50h]
  __int64 v46; // [rsp+78h] [rbp-48h]
  __int64 v47; // [rsp+80h] [rbp-40h]
  __int64 v48; // [rsp+88h] [rbp-38h]

  v5 = sub_BC0510(a4, &unk_4F86A90, a3);
  v6 = *a2;
  v7 = v5;
  v8 = *(__m128i **)(*a2 + 32);
  if ( *(_QWORD *)(*a2 + 24) - (_QWORD)v8 <= 0x21u )
  {
    sub_CB6200(v6, "SCCs for the program in PostOrder:", 0x22u);
  }
  else
  {
    si128 = _mm_load_si128((const __m128i *)&xmmword_3F70840);
    v8[2].m128i_i16[0] = 14962;
    *v8 = si128;
    v8[1] = _mm_load_si128((const __m128i *)&xmmword_3F70850);
    *(_QWORD *)(v6 + 32) += 34LL;
  }
  v10 = *(_QWORD *)(v7 + 64);
  v36[0] = 0;
  v36[1] = 0;
  v37 = 0;
  v38 = 0;
  v39 = 0;
  v40 = 0;
  v41 = 0;
  v42 = 0;
  v43 = 0;
  v44 = 0;
  v45 = 0;
  v46 = 0;
  v47 = 0;
  v48 = 0;
  sub_D126D0((__int64)v36, v10);
  sub_D12BD0((__int64)v36);
  v33 = 0;
  if ( v44 != v43 )
  {
    while ( 1 )
    {
      v11 = *a2;
      v12 = *(_QWORD *)(*a2 + 32);
      if ( (unsigned __int64)(*(_QWORD *)(*a2 + 24) - v12) <= 5 )
      {
        v11 = sub_CB6200(v11, "\nSCC #", 6u);
      }
      else
      {
        *(_DWORD *)v12 = 1128485642;
        *(_WORD *)(v12 + 4) = 8992;
        *(_QWORD *)(v11 + 32) += 6LL;
      }
      v13 = sub_CB59D0(v11, (unsigned int)++v33);
      v14 = *(_WORD **)(v13 + 32);
      if ( *(_QWORD *)(v13 + 24) - (_QWORD)v14 <= 1u )
      {
        sub_CB6200(v13, (unsigned __int8 *)": ", 2u);
      }
      else
      {
        *v14 = 8250;
        *(_QWORD *)(v13 + 32) += 2LL;
      }
      v15 = v44;
      if ( v44 != v43 )
        break;
LABEL_19:
      sub_D12BD0((__int64)v36);
      if ( v43 == v44 )
        goto LABEL_20;
    }
    v16 = *a2;
    v17 = (__int64 *)(v43 + 8);
    v18 = *(_QWORD *)(*(_QWORD *)v43 + 8LL);
    if ( !v18 )
      goto LABEL_16;
LABEL_10:
    v19 = sub_BD5D20(v18);
    v21 = *(void **)(v16 + 32);
    v22 = (unsigned __int8 *)v19;
    if ( *(_QWORD *)(v16 + 24) - (_QWORD)v21 < v20 )
    {
LABEL_29:
      sub_CB6200(v16, v22, v20);
    }
    else if ( v20 )
    {
      while ( 1 )
      {
        v34 = v20;
        memcpy(v21, v22, v20);
        *(_QWORD *)(v16 + 32) += v34;
        if ( v15 == (char *)v17 )
          break;
LABEL_13:
        v23 = *a2;
        v24 = *v17;
        v25 = *(_WORD **)(*a2 + 32);
        if ( *(_QWORD *)(*a2 + 24) - (_QWORD)v25 <= 1u )
        {
          v35 = *v17;
          sub_CB6200(v23, (unsigned __int8 *)", ", 2u);
          v24 = v35;
        }
        else
        {
          *v25 = 8236;
          *(_QWORD *)(v23 + 32) += 2LL;
        }
        v18 = *(_QWORD *)(v24 + 8);
        v16 = *a2;
        ++v17;
        if ( v18 )
          goto LABEL_10;
LABEL_16:
        v21 = *(void **)(v16 + 32);
        v20 = 13;
        v22 = "external node";
        if ( *(_QWORD *)(v16 + 24) - (_QWORD)v21 <= 0xCu )
          goto LABEL_29;
      }
LABEL_18:
      if ( v44 - v43 == 8 && (unsigned __int8)sub_D10A50((__int64)v36) )
      {
        v29 = *a2;
        v30 = *(__m128i **)(*a2 + 32);
        if ( *(_QWORD *)(*a2 + 24) - (_QWORD)v30 <= 0x10u )
        {
          sub_CB6200(v29, " (Has self-loop).", 0x11u);
        }
        else
        {
          v31 = _mm_load_si128((const __m128i *)&xmmword_3F70860);
          v30[1].m128i_i8[0] = 46;
          *v30 = v31;
          *(_QWORD *)(v29 + 32) += 17LL;
        }
      }
      goto LABEL_19;
    }
    if ( v15 != (char *)v17 )
      goto LABEL_13;
    goto LABEL_18;
  }
LABEL_20:
  if ( v46 )
    j_j___libc_free_0(v46, v48 - v46);
  if ( v43 )
    j_j___libc_free_0(v43, v45 - (_QWORD)v43);
  if ( v40 )
    j_j___libc_free_0(v40, v42 - v40);
  sub_C7D6A0(v37, 16LL * (unsigned int)v39, 8);
  v26 = *a2;
  v27 = *(_BYTE **)(*a2 + 32);
  if ( *(_BYTE **)(*a2 + 24) == v27 )
  {
    sub_CB6200(v26, (unsigned __int8 *)"\n", 1u);
  }
  else
  {
    *v27 = 10;
    ++*(_QWORD *)(v26 + 32);
  }
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 8) = a1 + 32;
  *(_QWORD *)(a1 + 56) = a1 + 80;
  *(_QWORD *)(a1 + 16) = 0x100000002LL;
  *(_QWORD *)(a1 + 64) = 2;
  *(_QWORD *)(a1 + 32) = &unk_4F82400;
  *(_DWORD *)(a1 + 72) = 0;
  *(_BYTE *)(a1 + 76) = 1;
  *(_DWORD *)(a1 + 24) = 0;
  *(_BYTE *)(a1 + 28) = 1;
  *(_QWORD *)a1 = 1;
  return a1;
}
