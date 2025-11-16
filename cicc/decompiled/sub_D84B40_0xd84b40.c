// Function: sub_D84B40
// Address: 0xd84b40
//
__int64 __fastcall sub_D84B40(__int64 a1, __int64 *a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rax
  __int64 v7; // r15
  _QWORD *v8; // r8
  void *v9; // rdx
  __int64 v10; // r14
  __m128i *v11; // rdi
  unsigned __int64 v12; // rbx
  unsigned __int8 *v13; // rsi
  unsigned __int64 v14; // rax
  __m128i si128; // xmm0
  _QWORD *v16; // rbx
  __int64 v17; // r13
  __int64 v18; // r15
  unsigned __int8 *v19; // rax
  size_t v20; // rdx
  void *v21; // rdi
  __int64 v22; // rdi
  _BYTE *v23; // rax
  __int64 v25; // rdi
  void *v26; // rdx
  __int64 v27; // rdi
  void *v28; // rdx
  __int64 v29; // rax
  __m128i *v30; // rbx
  __int64 v31; // rax
  __int64 v32; // rax
  __int64 v33; // [rsp+8h] [rbp-78h]
  size_t v34; // [rsp+10h] [rbp-70h]
  __int64 v35; // [rsp+20h] [rbp-60h]
  _QWORD *v37; // [rsp+28h] [rbp-58h]
  _QWORD *v38; // [rsp+28h] [rbp-58h]
  _QWORD *v39; // [rsp+28h] [rbp-58h]
  _QWORD *v40; // [rsp+28h] [rbp-58h]
  unsigned __int64 v41; // [rsp+30h] [rbp-50h] BYREF
  char v42; // [rsp+40h] [rbp-40h]

  v6 = sub_BC0510(a4, &unk_4F87C68, a3);
  v7 = *a2;
  v8 = (_QWORD *)a3;
  v35 = v6;
  v9 = *(void **)(*a2 + 32);
  v10 = v6 + 8;
  if ( *(_QWORD *)(*a2 + 24) - (_QWORD)v9 <= 0xCu )
  {
    v32 = sub_CB6200(v7, "Functions in ", 0xDu);
    v8 = (_QWORD *)a3;
    v11 = *(__m128i **)(v32 + 32);
    v7 = v32;
  }
  else
  {
    qmemcpy(v9, "Functions in ", 13);
    v11 = (__m128i *)(*(_QWORD *)(v7 + 32) + 13LL);
    *(_QWORD *)(v7 + 32) = v11;
  }
  v12 = v8[22];
  v13 = (unsigned __int8 *)v8[21];
  v14 = *(_QWORD *)(v7 + 24) - (_QWORD)v11;
  if ( v12 > v14 )
  {
    v40 = v8;
    v31 = sub_CB6200(v7, v13, v8[22]);
    v8 = v40;
    v11 = *(__m128i **)(v31 + 32);
    v7 = v31;
    v14 = *(_QWORD *)(v31 + 24) - (_QWORD)v11;
  }
  else if ( v12 )
  {
    v38 = v8;
    memcpy(v11, v13, v8[22]);
    v29 = *(_QWORD *)(v7 + 24);
    v30 = (__m128i *)(*(_QWORD *)(v7 + 32) + v12);
    *(_QWORD *)(v7 + 32) = v30;
    v8 = v38;
    v11 = v30;
    if ( (unsigned __int64)(v29 - (_QWORD)v30) > 0x1C )
      goto LABEL_6;
    goto LABEL_31;
  }
  if ( v14 > 0x1C )
  {
LABEL_6:
    si128 = _mm_load_si128((const __m128i *)&xmmword_3F72A90);
    qmemcpy(&v11[1], "nnotations: \n", 13);
    *v11 = si128;
    *(_QWORD *)(v7 + 32) += 29LL;
    goto LABEL_7;
  }
LABEL_31:
  v39 = v8;
  sub_CB6200(v7, " with hot/cold annotations: \n", 0x1Du);
  v8 = v39;
LABEL_7:
  v16 = (_QWORD *)v8[4];
  v37 = v8 + 3;
  if ( v16 != v8 + 3 )
  {
    v33 = a1;
    do
    {
      while ( 1 )
      {
        v17 = 0;
        v18 = *a2;
        if ( v16 )
          v17 = (__int64)(v16 - 7);
        v19 = (unsigned __int8 *)sub_BD5D20(v17);
        v21 = *(void **)(v18 + 32);
        if ( *(_QWORD *)(v18 + 24) - (_QWORD)v21 < v20 )
        {
          sub_CB6200(v18, v19, v20);
        }
        else if ( v20 )
        {
          v34 = v20;
          memcpy(v21, v19, v20);
          *(_QWORD *)(v18 + 32) += v34;
        }
        if ( v17 && *(_QWORD *)(v35 + 16) && (sub_B2EE70((__int64)&v41, v17, 0), v42) && sub_D84440(v10, v41) )
        {
          v25 = *a2;
          v26 = *(void **)(*a2 + 32);
          if ( *(_QWORD *)(*a2 + 24) - (_QWORD)v26 <= 0xBu )
          {
            sub_CB6200(v25, " :hot entry ", 0xCu);
          }
          else
          {
            qmemcpy(v26, " :hot entry ", 12);
            *(_QWORD *)(v25 + 32) += 12LL;
          }
        }
        else if ( sub_D84460(v10, v17) )
        {
          v27 = *a2;
          v28 = *(void **)(*a2 + 32);
          if ( *(_QWORD *)(*a2 + 24) - (_QWORD)v28 <= 0xCu )
          {
            sub_CB6200(v27, " :cold entry ", 0xDu);
          }
          else
          {
            qmemcpy(v28, " :cold entry ", 13);
            *(_QWORD *)(v27 + 32) += 13LL;
          }
        }
        v22 = *a2;
        v23 = *(_BYTE **)(*a2 + 32);
        if ( *(_BYTE **)(*a2 + 24) == v23 )
          break;
        *v23 = 10;
        ++*(_QWORD *)(v22 + 32);
        v16 = (_QWORD *)v16[1];
        if ( v37 == v16 )
          goto LABEL_21;
      }
      sub_CB6200(v22, (unsigned __int8 *)"\n", 1u);
      v16 = (_QWORD *)v16[1];
    }
    while ( v37 != v16 );
LABEL_21:
    a1 = v33;
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
