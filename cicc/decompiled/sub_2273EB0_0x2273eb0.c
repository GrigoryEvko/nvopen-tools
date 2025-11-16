// Function: sub_2273EB0
// Address: 0x2273eb0
//
__int64 __fastcall sub_2273EB0(__int64 a1, _QWORD *a2)
{
  unsigned int v2; // r13d
  __int64 v3; // r8
  __int64 v5; // rcx
  _QWORD *v7; // rax
  __m128i *v8; // rdx
  __int64 v9; // r13
  __m128i si128; // xmm0
  void *v11; // rdi
  _QWORD *v12; // r14
  _QWORD *v13; // r12
  _QWORD *v14; // r14
  _QWORD *v15; // r12
  __int64 v16; // rax
  unsigned __int8 *v17; // rsi
  size_t v18; // r12
  unsigned __int64 v19; // rax
  unsigned __int64 v20; // rax
  const char *v21; // rsi
  __int64 v22; // rax
  __m128i *v23; // rdx
  __m128i v24; // xmm0
  __int64 v25; // rax
  __int64 v26; // rax
  unsigned __int64 v27; // [rsp+0h] [rbp-D0h] BYREF
  unsigned __int64 v28; // [rsp+8h] [rbp-C8h] BYREF
  unsigned __int8 *v29[2]; // [rsp+10h] [rbp-C0h] BYREF
  __int64 v30; // [rsp+20h] [rbp-B0h] BYREF
  unsigned __int64 v31[2]; // [rsp+30h] [rbp-A0h] BYREF
  _BYTE v32[48]; // [rsp+40h] [rbp-90h] BYREF
  int v33; // [rsp+70h] [rbp-60h]
  _QWORD *v34; // [rsp+78h] [rbp-58h]
  _QWORD *v35; // [rsp+80h] [rbp-50h]
  __int64 v36; // [rsp+88h] [rbp-48h]
  _QWORD *v37; // [rsp+90h] [rbp-40h]
  _QWORD *v38; // [rsp+98h] [rbp-38h]
  __int64 v39; // [rsp+A0h] [rbp-30h]

  v2 = 0;
  v3 = a2[18];
  if ( v3 )
  {
    v5 = a2[17];
    v33 = 0;
    v31[1] = 0x600000000LL;
    v31[0] = (unsigned __int64)v32;
    v34 = 0;
    v35 = 0;
    v36 = 0;
    v37 = 0;
    v38 = 0;
    v39 = 0;
    sub_235CD80(&v27, a1, v31, v5, v3);
    if ( (v27 & 0xFFFFFFFFFFFFFFFELL) != 0 )
    {
      v27 = v27 & 0xFFFFFFFFFFFFFFFELL | 1;
      v7 = sub_CB72A0();
      v8 = (__m128i *)v7[4];
      v9 = (__int64)v7;
      if ( v7[3] - (_QWORD)v8 <= 0x10u )
      {
        v16 = sub_CB6200((__int64)v7, "Could not parse -", 0x11u);
        v11 = *(void **)(v16 + 32);
        v9 = v16;
      }
      else
      {
        si128 = _mm_load_si128((const __m128i *)&xmmword_4365C60);
        v8[1].m128i_i8[0] = 45;
        *v8 = si128;
        v11 = (void *)(v7[4] + 17LL);
        v7[4] = v11;
      }
      v17 = (unsigned __int8 *)a2[3];
      v18 = a2[4];
      v19 = *(_QWORD *)(v9 + 24) - (_QWORD)v11;
      if ( v19 < v18 )
      {
        v25 = sub_CB6200(v9, v17, v18);
        v11 = *(void **)(v25 + 32);
        v9 = v25;
        v19 = *(_QWORD *)(v25 + 24) - (_QWORD)v11;
      }
      else if ( v18 )
      {
        memcpy(v11, v17, v18);
        v26 = *(_QWORD *)(v9 + 24);
        v11 = (void *)(v18 + *(_QWORD *)(v9 + 32));
        *(_QWORD *)(v9 + 32) = v11;
        v19 = v26 - (_QWORD)v11;
      }
      if ( v19 <= 0xA )
      {
        v9 = sub_CB6200(v9, " pipeline: ", 0xBu);
      }
      else
      {
        qmemcpy(v11, " pipeline: ", 11);
        *(_QWORD *)(v9 + 32) += 11LL;
      }
      v20 = v27;
      v27 = 0;
      v28 = v20 | 1;
      sub_C64870((__int64)v29, (__int64 *)&v28);
      v21 = (const char *)v29[0];
      v22 = sub_CB6200(v9, v29[0], (size_t)v29[1]);
      v23 = *(__m128i **)(v22 + 32);
      if ( *(_QWORD *)(v22 + 24) - (_QWORD)v23 <= 0x1Bu )
      {
        v21 = "... I'm going to ignore it.\n";
        sub_CB6200(v22, "... I'm going to ignore it.\n", 0x1Cu);
      }
      else
      {
        v24 = _mm_load_si128((const __m128i *)&xmmword_4365C70);
        qmemcpy(&v23[1], " ignore it.\n", 12);
        *v23 = v24;
        *(_QWORD *)(v22 + 32) += 28LL;
      }
      if ( (__int64 *)v29[0] != &v30 )
      {
        v21 = (const char *)(v30 + 1);
        j_j___libc_free_0((unsigned __int64)v29[0]);
      }
      if ( (v28 & 1) != 0 || (v28 & 0xFFFFFFFFFFFFFFFELL) != 0 )
        sub_C63C30(&v28, (__int64)v21);
      if ( (v27 & 1) != 0 || (v2 = 0, (v27 & 0xFFFFFFFFFFFFFFFELL) != 0) )
        sub_C63C30(&v27, (__int64)v21);
    }
    else
    {
      v2 = 1;
    }
    v12 = v38;
    v13 = v37;
    if ( v38 != v37 )
    {
      do
      {
        if ( *v13 )
          (*(void (__fastcall **)(_QWORD))(*(_QWORD *)*v13 + 8LL))(*v13);
        ++v13;
      }
      while ( v12 != v13 );
      v13 = v37;
    }
    if ( v13 )
      j_j___libc_free_0((unsigned __int64)v13);
    v14 = v35;
    v15 = v34;
    if ( v35 != v34 )
    {
      do
      {
        if ( *v15 )
          (*(void (__fastcall **)(_QWORD))(*(_QWORD *)*v15 + 8LL))(*v15);
        ++v15;
      }
      while ( v14 != v15 );
      v15 = v34;
    }
    if ( v15 )
      j_j___libc_free_0((unsigned __int64)v15);
    if ( (_BYTE *)v31[0] != v32 )
      _libc_free(v31[0]);
  }
  return v2;
}
