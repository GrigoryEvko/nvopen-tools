// Function: sub_2C76F10
// Address: 0x2c76f10
//
struct __jmp_buf_tag *__fastcall sub_2C76F10(__int64 a1, char *a2, __int64 a3)
{
  char *v4; // rsi
  __int64 v6; // rax
  __m128i *v7; // rdx
  __int64 v8; // r12
  __m128i si128; // xmm0
  size_t v10; // rax
  char *v11; // rcx
  _BYTE *v12; // rdi
  size_t v13; // r14
  _BYTE *v14; // rax
  __int64 v15; // rdx
  _BYTE *v16; // rax
  struct __jmp_buf_tag *result; // rax
  __int64 v18; // rdi

  v4 = (char *)a3;
  v6 = sub_2C76A00(a1, a3, 0);
  v7 = *(__m128i **)(v6 + 32);
  v8 = v6;
  if ( *(_QWORD *)(v6 + 24) - (_QWORD)v7 <= 0x14u )
  {
    v4 = "Illegal instruction: ";
    v8 = sub_CB6200(v6, "Illegal instruction: ", 0x15u);
  }
  else
  {
    si128 = _mm_load_si128((const __m128i *)&xmmword_42D0530);
    v7[1].m128i_i32[0] = 980316009;
    v7[1].m128i_i8[4] = 32;
    *v7 = si128;
    *(_QWORD *)(v6 + 32) += 21LL;
  }
  v10 = strlen(a2);
  v12 = *(_BYTE **)(v8 + 32);
  v13 = v10;
  v14 = *(_BYTE **)(v8 + 24);
  v15 = v14 - v12;
  if ( v13 > v14 - v12 )
  {
    v4 = a2;
    v8 = sub_CB6200(v8, (unsigned __int8 *)a2, v13);
    v12 = *(_BYTE **)(v8 + 32);
    if ( *(_BYTE **)(v8 + 24) != v12 )
    {
LABEL_7:
      *v12 = 10;
      ++*(_QWORD *)(v8 + 32);
      goto LABEL_8;
    }
  }
  else
  {
    if ( v13 )
    {
      v4 = a2;
      memcpy(v12, a2, v13);
      v14 = *(_BYTE **)(v8 + 24);
      v12 = (_BYTE *)(v13 + *(_QWORD *)(v8 + 32));
      *(_QWORD *)(v8 + 32) = v12;
    }
    if ( v14 != v12 )
      goto LABEL_7;
  }
  v4 = "\n";
  sub_CB6200(v8, (unsigned __int8 *)"\n", 1u);
LABEL_8:
  v16 = *(_BYTE **)(a1 + 16);
  if ( v16 )
    *v16 = 0;
  result = (struct __jmp_buf_tag *)*(unsigned int *)(a1 + 4);
  if ( !(_DWORD)result )
  {
    v18 = *(_QWORD *)(a1 + 24);
    if ( *(_QWORD *)(v18 + 32) != *(_QWORD *)(v18 + 16) )
    {
      sub_CB5AE0((__int64 *)v18);
      v18 = *(_QWORD *)(a1 + 24);
    }
    return sub_CEB520(*(_QWORD **)(v18 + 48), (__int64)v4, v15, v11);
  }
  return result;
}
