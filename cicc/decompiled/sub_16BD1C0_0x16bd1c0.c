// Function: sub_16BD1C0
// Address: 0x16bd1c0
//
void __fastcall sub_16BD1C0(const char *src, unsigned __int8 a2)
{
  __int64 v2; // rbx
  __int64 v3; // r12
  __int64 v4; // r13
  __int64 v5; // r14
  __int64 v6; // r15
  unsigned int v8; // eax
  void (__fastcall *v9)(__int64, __m128i *, _QWORD, __int64, size_t); // rbx
  __int64 v10; // r14
  size_t v11; // rax
  __int64 v12; // rcx
  size_t v13; // r8
  _QWORD *v14; // rdx
  __int64 v15; // rax
  _QWORD *v16; // rdi
  size_t n; // [rsp+0h] [rbp-70h]
  size_t v18; // [rsp+18h] [rbp-58h] BYREF
  __m128i buf; // [rsp+20h] [rbp-50h] BYREF
  _QWORD v20[8]; // [rsp+30h] [rbp-40h] BYREF

  v20[7] = v6;
  v20[6] = v5;
  v20[5] = v4;
  v20[4] = v3;
  v20[3] = v2;
  if ( &_pthread_key_create )
  {
    v8 = pthread_mutex_lock(&stru_4FA0360);
    if ( v8 )
      sub_4264C5(v8);
  }
  v9 = (void (__fastcall *)(__int64, __m128i *, _QWORD, __int64, size_t))qword_4FA03D0;
  v10 = qword_4FA03C8;
  if ( &_pthread_key_create )
    pthread_mutex_unlock(&stru_4FA0360);
  if ( !v9 )
  {
    strcpy((char *)v20, "of memory\n");
    buf = _mm_load_si128((const __m128i *)&xmmword_42AEDF0);
    write(2, &buf, 0x1Au);
    abort();
  }
  buf.m128i_i64[0] = (__int64)v20;
  if ( !src )
    sub_426248((__int64)"basic_string::_M_construct null not valid");
  v11 = strlen(src);
  v18 = v11;
  v13 = v11;
  if ( v11 > 0xF )
  {
    n = v11;
    v15 = sub_22409D0(&buf, &v18, 0);
    v13 = n;
    buf.m128i_i64[0] = v15;
    v16 = (_QWORD *)v15;
    v20[0] = v18;
LABEL_14:
    memcpy(v16, src, v13);
    v11 = v18;
    v14 = (_QWORD *)buf.m128i_i64[0];
    goto LABEL_10;
  }
  if ( v11 == 1 )
  {
    LOBYTE(v20[0]) = *src;
    v14 = v20;
    goto LABEL_10;
  }
  if ( v11 )
  {
    v16 = v20;
    goto LABEL_14;
  }
  v14 = v20;
LABEL_10:
  buf.m128i_i64[1] = v11;
  *((_BYTE *)v14 + v11) = 0;
  v9(v10, &buf, a2, v12, v13);
  if ( (_QWORD *)buf.m128i_i64[0] != v20 )
    j_j___libc_free_0(buf.m128i_i64[0], v20[0] + 1LL);
  def_16BF26D();
}
