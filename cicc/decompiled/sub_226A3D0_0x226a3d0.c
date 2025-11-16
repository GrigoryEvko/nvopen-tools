// Function: sub_226A3D0
// Address: 0x226a3d0
//
_QWORD *__fastcall sub_226A3D0(
        _QWORD *a1,
        _QWORD *a2,
        __int64 *a3,
        __m128i *a4,
        __int64 a5,
        unsigned __int64 *a6,
        __m128i a7,
        __int64 a8,
        _QWORD *a9)
{
  __int64 *v11; // r13
  struct __jmp_buf_tag *v12; // r12
  int v13; // eax
  char v14; // bl
  char *v15; // rax
  char v16; // al
  __int64 v17; // rsi
  int *v18; // rax
  int v19; // eax
  _QWORD *v20; // rdx
  int v21; // ebx
  __int64 v22; // rax
  __m128i v23; // xmm0
  __int64 v24; // rdx
  char *v25; // rcx
  const void *v26; // rsi
  __int64 v27; // r13
  int v28; // r8d
  __int64 v29; // r12
  _DWORD *v30; // rax
  char v31; // bl
  _OWORD *v32; // rax
  __m128i si128; // xmm0
  char *v34; // rsi
  size_t v35; // rdx
  __int64 v36; // r15
  _DWORD *v37; // rax
  _DWORD *v38; // rax
  void *v39; // rdi
  __int64 v40; // r12
  __int64 v41; // rax
  __int64 v42; // rbx
  unsigned __int64 v43; // rdi
  int v44; // r12d
  __int64 v51; // [rsp+30h] [rbp-780h] BYREF
  unsigned __int64 v52; // [rsp+38h] [rbp-778h] BYREF
  _QWORD *v53; // [rsp+40h] [rbp-770h] BYREF
  __int64 v54; // [rsp+48h] [rbp-768h]
  _QWORD v55[2]; // [rsp+50h] [rbp-760h] BYREF
  unsigned __int64 v56[2]; // [rsp+60h] [rbp-750h] BYREF
  _QWORD v57[2]; // [rsp+70h] [rbp-740h] BYREF
  _DWORD v58[460]; // [rsp+80h] [rbp-730h] BYREF

  if ( (unsigned __int8)sub_2C72F80() )
  {
    sub_CEAF80(a3);
    return 0;
  }
  v11 = sub_CEACC0();
  v12 = (struct __jmp_buf_tag *)sub_C94E20((__int64)v11);
  if ( !v12 )
  {
    v39 = (void *)sub_CEECD0(200, 8u);
    memset(v39, 0, 0xC8u);
    sub_C94E10((__int64)v11, v39);
    v12 = (struct __jmp_buf_tag *)sub_C94E20((__int64)v11);
  }
  v13 = _setjmp(v12);
  if ( v13 )
  {
    if ( v13 == 1 )
    {
      sub_CEAF80(a3);
      return 0;
    }
    goto LABEL_23;
  }
  v26 = (const void *)a1[36];
  if ( !v26 )
  {
    v40 = a1[35];
    v41 = sub_22077B0(0x6F8u);
    v42 = v41;
    if ( v41 )
    {
      sub_22623C0(v41, v40);
      *(_QWORD *)(v42 + 1776) = 0;
    }
    v43 = a1[36];
    a1[36] = v42;
    if ( v43 )
    {
      j_j___libc_free_0(v43);
      v42 = a1[36];
    }
    *(_QWORD *)(v42 + 1776) = a1[37];
    v26 = (const void *)a1[36];
  }
  qmemcpy(v58, v26, 0x6F8u);
  v27 = (__int64)(a1 + 1);
  v28 = v58[390];
  v14 = v58[232];
  if ( v58[390] < 0 )
    v28 = v58[400];
  v44 = v28;
  if ( LOBYTE(v58[232]) )
  {
    v14 = 0;
  }
  else if ( !LOBYTE(v58[242]) )
  {
    v14 = LOBYTE(v58[252]) ^ 1;
  }
  v15 = (char *)sub_C94E20((__int64)qword_4F86470);
  if ( v15 )
    v16 = *v15;
  else
    v16 = qword_4F86470[2];
  if ( v16 || v14 )
  {
    if ( sub_C96F30() )
      goto LABEL_32;
  }
  else
  {
    if ( !v44 )
      v44 = sub_22420F0();
    if ( sub_C96F30() )
    {
      if ( v44 > 1 )
      {
        v17 = a1[32];
        v18 = (int *)a1[31];
LABEL_19:
        v19 = sub_226A0F0(*v18, v17, a2, v27, v44, (__int64)v58, a7, a4, a5, a6, a8, a9);
        a2 = v20;
        v21 = v19;
        goto LABEL_20;
      }
LABEL_32:
      v29 = sub_C996C0("Phase I", 7, 0, 0);
      v30 = (_DWORD *)sub_CEECD0(4, 4u);
      *v30 = 1;
      sub_C94E10((__int64)qword_4F86310, v30);
      sub_226C400(*(_DWORD *)a1[31], a1[32], (_DWORD)a2, (unsigned int)v58, v27, *a9, a9[1]);
      if ( *a9 && ((unsigned int (__fastcall *)(_QWORD, _QWORD))*a9)(a9[1], 0) )
      {
        if ( v29 )
          sub_C9AF60(v29);
        return a2;
      }
      v51 = 27;
      v31 = sub_2260560(a2, (__int64)v58);
      v53 = v55;
      v32 = (_OWORD *)sub_22409D0((__int64)&v53, (unsigned __int64 *)&v51, 0);
      si128 = _mm_load_si128((const __m128i *)&xmmword_4281B30);
      v53 = v32;
      v34 = "Yes";
      v55[0] = v51;
      qmemcpy(v32 + 1, "Concurrent=", 11);
      *v32 = si128;
      v54 = v51;
      *((_BYTE *)v53 + v51) = 0;
      v35 = 3LL - (v31 == 0);
      if ( !v31 )
        v34 = "No";
      if ( v35 > 0x3FFFFFFFFFFFFFFFLL - v54 )
        sub_4262D8((__int64)"basic_string::append");
      sub_2241490((unsigned __int64 *)&v53, v34, v35);
      v36 = sub_C996C0("Phase II", 8, (__int64)v53, v54);
      v37 = (_DWORD *)sub_CEECD0(4, 4u);
      *v37 = 2;
      sub_C94E10((__int64)qword_4F86310, v37);
      v21 = sub_226C400(*(_DWORD *)a1[31], a1[32], (_DWORD)a2, (unsigned int)v58, v27, *a9, a9[1]);
      v38 = (_DWORD *)sub_CEECD0(4, 4u);
      *v38 = 3;
      sub_C94E10((__int64)qword_4F86310, v38);
      if ( v36 )
        sub_C9AF60(v36);
      if ( v53 != v55 )
        j_j___libc_free_0((unsigned __int64)v53);
      if ( v29 )
        sub_C9AF60(v29);
      goto LABEL_20;
    }
    v17 = a1[32];
    v18 = (int *)a1[31];
    if ( v44 > 1 )
      goto LABEL_19;
  }
  v21 = sub_226C400(*(_DWORD *)a1[31], a1[32], (_DWORD)a2, (unsigned int)v58, v27, *a9, a9[1]);
LABEL_20:
  if ( v21 )
  {
    v52 = 33;
    v56[0] = (unsigned __int64)v57;
    v22 = sub_22409D0((__int64)v56, &v52, 0);
    v56[0] = v22;
    v57[0] = v52;
    *(__m128i *)v22 = _mm_load_si128((const __m128i *)&xmmword_4363B50);
    v23 = _mm_load_si128((const __m128i *)&xmmword_4363B60);
    v24 = v56[0];
    *(_BYTE *)(v22 + 32) = 103;
    *(__m128i *)(v22 + 16) = v23;
    v56[1] = v52;
    *(_BYTE *)(v24 + v52) = 0;
    sub_CEB520(v56, (__int64)&v52, v24, v25);
    if ( (_QWORD *)v56[0] != v57 )
      j_j___libc_free_0(v56[0]);
  }
LABEL_23:
  sub_CEAF80(a3);
  return a2;
}
