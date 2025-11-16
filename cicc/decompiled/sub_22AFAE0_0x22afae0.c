// Function: sub_22AFAE0
// Address: 0x22afae0
//
__int64 __fastcall sub_22AFAE0(__int64 a1, __int64 *a2, __int64 a3, __int64 a4)
{
  __int64 v5; // rax
  __int64 v6; // rax
  __m128i *v7; // rdx
  __int64 v8; // rdi
  __m128i si128; // xmm0
  __int64 v10; // rax
  void *v11; // rdx
  __int64 v12; // rbx
  __int64 v13; // rdx
  __int64 v14; // rdi
  __m128i *v15; // rdx
  __m128i v16; // xmm0
  __int64 v17; // rdi
  __m128i *v18; // rdx
  __m128i v19; // xmm0
  __int64 v20; // rdi
  _BYTE *v21; // rax
  __int64 v22; // r14
  void *v23; // rdx
  __int64 v24; // rax
  char *v25; // rax
  __int64 v26; // rdx
  size_t v27; // rdx
  unsigned __int8 *v28; // rsi
  __int64 v29; // rax
  void *v30; // rdx
  char *v31; // rax
  __int64 v32; // rdx
  int v33; // eax
  __int64 v34; // r14
  char *v35; // rax
  __int64 v36; // rdx
  size_t v37; // rdx
  unsigned __int8 *v38; // rsi
  _QWORD *v41; // [rsp+8h] [rbp-78h]
  _QWORD *i; // [rsp+10h] [rbp-70h]
  int v43; // [rsp+24h] [rbp-5Ch]
  __int64 v44; // [rsp+28h] [rbp-58h]
  unsigned __int8 *v45; // [rsp+30h] [rbp-50h] BYREF
  size_t v46; // [rsp+38h] [rbp-48h]
  _BYTE v47[64]; // [rsp+40h] [rbp-40h] BYREF

  v5 = sub_BC0510(a4, &unk_4FDB950, a3);
  v41 = *(_QWORD **)(v5 + 328);
  for ( i = *(_QWORD **)(v5 + 320); v41 != i; i += 3 )
  {
    v6 = sub_CB59D0(*a2, 0x86BCA1AF286BCA1BLL * ((__int64)(i[1] - *i) >> 3));
    v7 = *(__m128i **)(v6 + 32);
    v8 = v6;
    if ( *(_QWORD *)(v6 + 24) - (_QWORD)v7 <= 0x15u )
    {
      v8 = sub_CB6200(v6, " candidates of length ", 0x16u);
    }
    else
    {
      si128 = _mm_load_si128((const __m128i *)&xmmword_4366A20);
      v7[1].m128i_i32[0] = 1952935525;
      v7[1].m128i_i16[2] = 8296;
      *v7 = si128;
      *(_QWORD *)(v6 + 32) += 22LL;
    }
    v10 = sub_CB59D0(v8, *(unsigned int *)(*i + 4LL));
    v11 = *(void **)(v10 + 32);
    if ( *(_QWORD *)(v10 + 24) - (_QWORD)v11 <= 0xDu )
    {
      sub_CB6200(v10, ".  Found in: \n", 0xEu);
    }
    else
    {
      qmemcpy(v11, ".  Found in: \n", 14);
      *(_QWORD *)(v10 + 32) += 14LL;
    }
    v12 = *i;
    v44 = i[1];
    if ( *i != v44 )
    {
      do
      {
        while ( 1 )
        {
          v22 = *a2;
          v23 = *(void **)(*a2 + 32);
          if ( *(_QWORD *)(*a2 + 24) - (_QWORD)v23 <= 0xBu )
          {
            v22 = sub_CB6200(*a2, "  Function: ", 0xCu);
          }
          else
          {
            qmemcpy(v23, "  Function: ", 12);
            *(_QWORD *)(v22 + 32) += 12LL;
          }
          v24 = sub_B43CB0(*(_QWORD *)(*(_QWORD *)(v12 + 8) + 16LL));
          v25 = (char *)sub_BD5D20(v24);
          v45 = v47;
          if ( v25 )
          {
            sub_22AD7C0((__int64 *)&v45, v25, (__int64)&v25[v26]);
            v27 = v46;
            v28 = v45;
          }
          else
          {
            v28 = v47;
            v46 = 0;
            v27 = 0;
            v47[0] = 0;
          }
          v29 = sub_CB6200(v22, v28, v27);
          v30 = *(void **)(v29 + 32);
          if ( *(_QWORD *)(v29 + 24) - (_QWORD)v30 <= 0xEu )
          {
            sub_CB6200(v29, ", Basic Block: ", 0xFu);
          }
          else
          {
            qmemcpy(v30, ", Basic Block: ", 15);
            *(_QWORD *)(v29 + 32) += 15LL;
          }
          if ( v45 != v47 )
            j_j___libc_free_0((unsigned __int64)v45);
          v31 = (char *)sub_BD5D20(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v12 + 8) + 16LL) + 40LL));
          v45 = v47;
          if ( v31 )
          {
            sub_22AD7C0((__int64 *)&v45, v31, (__int64)&v31[v32]);
          }
          else
          {
            v46 = 0;
            v47[0] = 0;
          }
          v33 = sub_2241AC0((__int64)&v45, byte_3F871B3);
          if ( v45 != v47 )
          {
            v43 = v33;
            j_j___libc_free_0((unsigned __int64)v45);
            v33 = v43;
          }
          v34 = *a2;
          if ( v33 )
          {
            v35 = (char *)sub_BD5D20(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v12 + 8) + 16LL) + 40LL));
            v45 = v47;
            if ( v35 )
            {
              sub_22AD7C0((__int64 *)&v45, v35, (__int64)&v35[v36]);
              v37 = v46;
              v38 = v45;
            }
            else
            {
              v38 = v47;
              v46 = 0;
              v37 = 0;
              v47[0] = 0;
            }
            sub_CB6200(v34, v38, v37);
            if ( v45 != v47 )
              j_j___libc_free_0((unsigned __int64)v45);
          }
          else
          {
            v13 = *(_QWORD *)(v34 + 32);
            if ( (unsigned __int64)(*(_QWORD *)(v34 + 24) - v13) <= 8 )
            {
              sub_CB6200(*a2, "(unnamed)", 9u);
            }
            else
            {
              *(_BYTE *)(v13 + 8) = 41;
              *(_QWORD *)v13 = 0x64656D616E6E7528LL;
              *(_QWORD *)(v34 + 32) += 9LL;
            }
          }
          v14 = *a2;
          v15 = *(__m128i **)(*a2 + 32);
          if ( *(_QWORD *)(*a2 + 24) - (_QWORD)v15 <= 0x17u )
          {
            sub_CB6200(v14, "\n    Start Instruction: ", 0x18u);
          }
          else
          {
            v16 = _mm_load_si128((const __m128i *)&xmmword_4366A30);
            v15[1].m128i_i64[0] = 0x203A6E6F69746375LL;
            *v15 = v16;
            *(_QWORD *)(v14 + 32) += 24LL;
          }
          sub_A69870(*(_QWORD *)(*(_QWORD *)(v12 + 8) + 16LL), (_BYTE *)*a2, 0);
          v17 = *a2;
          v18 = *(__m128i **)(*a2 + 32);
          if ( *(_QWORD *)(*a2 + 24) - (_QWORD)v18 <= 0x17u )
          {
            sub_CB6200(v17, "\n      End Instruction: ", 0x18u);
          }
          else
          {
            v19 = _mm_load_si128((const __m128i *)&xmmword_4366A40);
            v18[1].m128i_i64[0] = 0x203A6E6F69746375LL;
            *v18 = v19;
            *(_QWORD *)(v17 + 32) += 24LL;
          }
          sub_A69870(*(_QWORD *)(*(_QWORD *)(v12 + 16) + 16LL), (_BYTE *)*a2, 0);
          v20 = *a2;
          v21 = *(_BYTE **)(*a2 + 32);
          if ( *(_BYTE **)(*a2 + 24) == v21 )
            break;
          *v21 = 10;
          v12 += 152;
          ++*(_QWORD *)(v20 + 32);
          if ( v44 == v12 )
            goto LABEL_38;
        }
        v12 += 152;
        sub_CB6200(v20, (unsigned __int8 *)"\n", 1u);
      }
      while ( v44 != v12 );
    }
LABEL_38:
    ;
  }
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 8) = a1 + 32;
  *(_QWORD *)(a1 + 56) = a1 + 80;
  *(_QWORD *)(a1 + 16) = 0x100000002LL;
  *(_QWORD *)(a1 + 64) = 2;
  *(_QWORD *)(a1 + 32) = &qword_4F82400;
  *(_DWORD *)(a1 + 72) = 0;
  *(_BYTE *)(a1 + 76) = 1;
  *(_DWORD *)(a1 + 24) = 0;
  *(_BYTE *)(a1 + 28) = 1;
  *(_QWORD *)a1 = 1;
  return a1;
}
