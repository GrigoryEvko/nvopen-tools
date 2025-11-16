// Function: sub_1D4A180
// Address: 0x1d4a180
//
void *__fastcall sub_1D4A180(_QWORD *a1, __int64 a2)
{
  __int64 *v2; // rax
  struct __jmp_buf_tag *v3; // rax
  __int64 v4; // r8
  unsigned int *v5; // rdx
  __int64 v6; // rax
  _QWORD *v7; // r12
  int v8; // r9d
  __int64 v9; // rdi
  __int64 (*v10)(); // rax
  __m128i *v11; // rdx
  __m128i si128; // xmm0
  const char **v13; // rdi
  __m128i *v14; // rdx
  const char **v15; // r12
  void *result; // rax
  __m128i *v17; // rdx
  const char **v18; // r14
  __int64 v19; // rbx
  __m128i *v20; // rdx
  unsigned int v21; // r9d
  __m128i v22; // xmm0
  __int64 v23; // rax
  char *v24; // rax
  size_t v25; // rdx
  char *v26; // rdi
  __int64 v27; // rax
  __int64 *v28; // r12
  _BYTE *v29; // rax
  size_t v30; // [rsp+0h] [rbp-B0h]
  struct __jmp_buf_tag *env; // [rsp+8h] [rbp-A8h]
  _QWORD v32[2]; // [rsp+10h] [rbp-A0h] BYREF
  _QWORD v33[2]; // [rsp+20h] [rbp-90h] BYREF
  char *v34; // [rsp+30h] [rbp-80h] BYREF
  size_t v35; // [rsp+38h] [rbp-78h]
  _QWORD v36[2]; // [rsp+40h] [rbp-70h] BYREF
  const char *v37; // [rsp+50h] [rbp-60h] BYREF
  __m128i *v38; // [rsp+58h] [rbp-58h]
  __int64 v39; // [rsp+60h] [rbp-50h] BYREF
  __m128i *v40; // [rsp+68h] [rbp-48h]
  int v41; // [rsp+70h] [rbp-40h]
  _QWORD *v42; // [rsp+78h] [rbp-38h]

  v34 = (char *)v36;
  v35 = 0;
  LOBYTE(v36[0]) = 0;
  sub_1C315E0((__int64)&v37, (__int64 *)(a2 + 72));
  sub_2241490(&v34, v37, v38);
  if ( v37 != (const char *)&v39 )
    j_j___libc_free_0(v37, v39 + 1);
  if ( 0x3FFFFFFFFFFFFFFFLL - v35 <= 0x1C )
    sub_4262D8((__int64)"basic_string::append");
  sub_2241490(&v34, " Error: unsupported operation", 29);
  sub_1C3EFD0((__int64)&v34, 1);
  if ( v34 != (char *)v36 )
    j_j___libc_free_0(v34, v36[0] + 1LL);
  v2 = sub_1C3E710();
  v3 = (struct __jmp_buf_tag *)sub_16D40F0((__int64)v2);
  if ( v3 )
  {
    env = v3;
    v28 = sub_1C3E7B0();
    v29 = (_BYTE *)sub_1C42D70(1, 1);
    *v29 = 1;
    sub_16D40E0((__int64)v28, v29);
    longjmp(env, 1);
  }
  v42 = v32;
  v37 = (const char *)&unk_49EFBE0;
  v32[0] = v33;
  v32[1] = 0;
  LOBYTE(v33[0]) = 0;
  v41 = 1;
  v40 = 0;
  v39 = 0;
  v38 = 0;
  sub_16E7EE0((__int64)&v37, "Cannot select: ", 0xFu);
  if ( (unsigned __int16)(*(_WORD *)(a2 + 24) - 43) > 2u )
  {
    sub_20981D0(a2, &v37, a1[34]);
    v17 = v40;
    if ( (unsigned __int64)(v39 - (_QWORD)v40) <= 0xD )
    {
      v18 = (const char **)sub_16E7EE0((__int64)&v37, "\nIn function: ", 0xEu);
    }
    else
    {
      v40->m128i_i32[2] = 1852795252;
      v18 = &v37;
      v17->m128i_i64[0] = 0x636E7566206E490ALL;
      v17->m128i_i16[6] = 8250;
      v40 = (__m128i *)((char *)v40 + 14);
    }
    v24 = (char *)sub_1E0A440(a1[32]);
    v26 = (char *)v18[3];
    if ( v18[2] - v26 < v25 )
    {
      sub_16E7EE0((__int64)v18, v24, v25);
    }
    else if ( v25 )
    {
      v30 = v25;
      memcpy(v26, v24, v25);
      v18[3] += v30;
    }
  }
  else
  {
    v5 = *(unsigned int **)(a2 + 32);
    v6 = *(_QWORD *)(*(_QWORD *)&v5[10 * (*(_BYTE *)(*(_QWORD *)(*(_QWORD *)v5 + 40LL) + 16LL * v5[2]) == 1)] + 88LL);
    v7 = *(_QWORD **)(v6 + 24);
    if ( *(_DWORD *)(v6 + 32) > 0x40u )
      v7 = (_QWORD *)*v7;
    v8 = (int)v7;
    if ( (unsigned int)v7 <= 0x1DC9 )
    {
      v14 = v40;
      if ( (unsigned __int64)(v39 - (_QWORD)v40) <= 0xA )
      {
        v23 = sub_16E7EE0((__int64)&v37, "intrinsic %", 0xBu);
        v8 = (int)v7;
        v15 = (const char **)v23;
      }
      else
      {
        v40->m128i_i8[10] = 37;
        v15 = &v37;
        qmemcpy(v14, "intrinsic ", 10);
        v40 = (__m128i *)((char *)v40 + 11);
      }
      sub_15E1070((__int64 *)&v34, v8, 0, 0);
    }
    else
    {
      v9 = a1[29];
      v10 = *(__int64 (**)())(*(_QWORD *)v9 + 32LL);
      if ( v10 == sub_16FF770
        || (v19 = ((__int64 (__fastcall *)(__int64, const char *, unsigned int *, _QWORD, __int64, _QWORD))v10)(
                    v9,
                    "Cannot select: ",
                    v5,
                    *(_QWORD *)v5,
                    v4,
                    (unsigned int)v7)) == 0 )
      {
        v11 = v40;
        if ( (unsigned __int64)(v39 - (_QWORD)v40) <= 0x12 )
        {
          v13 = (const char **)sub_16E7EE0((__int64)&v37, "unknown intrinsic #", 0x13u);
        }
        else
        {
          si128 = _mm_load_si128((const __m128i *)&xmmword_42E82F0);
          v40[1].m128i_i8[2] = 35;
          v13 = &v37;
          v11[1].m128i_i16[0] = 8291;
          *v11 = si128;
          v40 = (__m128i *)((char *)v40 + 19);
        }
        sub_16E7A90((__int64)v13, (unsigned int)v7);
        goto LABEL_20;
      }
      v20 = v40;
      v21 = (unsigned int)v7;
      if ( (unsigned __int64)(v39 - (_QWORD)v40) <= 0x11 )
      {
        v27 = sub_16E7EE0((__int64)&v37, "target intrinsic %", 0x12u);
        v21 = (unsigned int)v7;
        v15 = (const char **)v27;
      }
      else
      {
        v22 = _mm_load_si128((const __m128i *)&xmmword_42E82E0);
        v15 = &v37;
        v40[1].m128i_i16[0] = 9504;
        *v20 = v22;
        v40 = (__m128i *)((char *)v40 + 18);
      }
      (*(void (__fastcall **)(char **, __int64, _QWORD, _QWORD, _QWORD))(*(_QWORD *)v19 + 16LL))(&v34, v19, v21, 0, 0);
    }
    sub_16E7EE0((__int64)v15, v34, v35);
    if ( v34 != (char *)v36 )
      j_j___libc_free_0(v34, v36[0] + 1LL);
  }
LABEL_20:
  if ( v40 != v38 )
    sub_16E7BA0((__int64 *)&v37);
  sub_1C3EF50((__int64)v42);
  result = sub_16E7BC0((__int64 *)&v37);
  if ( (_QWORD *)v32[0] != v33 )
    return (void *)j_j___libc_free_0(v32[0], v33[0] + 1LL);
  return result;
}
