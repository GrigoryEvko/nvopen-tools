// Function: sub_2396290
// Address: 0x2396290
//
__int64 *__fastcall sub_2396290(__int64 *a1, __m128i *a2, unsigned __int64 *a3, _BYTE *a4, unsigned __int64 a5)
{
  unsigned __int64 v6; // rax
  unsigned int v8; // eax
  unsigned int v9; // r14d
  __int64 v10; // rdx
  __int64 v11; // r13
  unsigned int v12; // eax
  unsigned int v13; // ebx
  __int64 v14; // rdx
  __int64 v15; // r14
  _QWORD v16[2]; // [rsp+0h] [rbp-F0h] BYREF
  __m128i v17; // [rsp+10h] [rbp-E0h] BYREF
  const __m128i *v18; // [rsp+20h] [rbp-D0h] BYREF
  const __m128i *v19; // [rsp+28h] [rbp-C8h]
  char v20; // [rsp+38h] [rbp-B8h]
  unsigned __int64 v21[2]; // [rsp+40h] [rbp-B0h] BYREF
  __int64 v22; // [rsp+50h] [rbp-A0h] BYREF
  const char *v23; // [rsp+60h] [rbp-90h] BYREF
  __int64 v24; // [rsp+68h] [rbp-88h]
  _QWORD *v25; // [rsp+70h] [rbp-80h]
  __int64 v26; // [rsp+78h] [rbp-78h]
  char v27; // [rsp+80h] [rbp-70h]
  void *v28; // [rsp+88h] [rbp-68h] BYREF
  _QWORD *v29; // [rsp+90h] [rbp-60h]
  _QWORD v30[2]; // [rsp+98h] [rbp-58h] BYREF
  _QWORD v31[9]; // [rsp+A8h] [rbp-48h] BYREF

  v16[0] = a4;
  v16[1] = a5;
  sub_2352D90((__int64)&v18, a4, a5);
  if ( !v20 || v18 == v19 )
  {
    v8 = sub_C63BB0();
    v24 = 22;
    v9 = v8;
    v11 = v10;
    v23 = "invalid pipeline '{0}'";
    v25 = v30;
    v26 = 1;
    v27 = 1;
    v28 = &unk_49DB108;
    v29 = v16;
    v30[0] = &v28;
    sub_23328D0((__int64)v21, (__int64)&v23);
    sub_23058C0(a1, (__int64)v21, v9, v11);
    if ( (__int64 *)v21[0] != &v22 )
    {
      j_j___libc_free_0(v21[0]);
      if ( !v20 )
        return a1;
LABEL_11:
      v20 = 0;
      sub_234A6B0((unsigned __int64 *)&v18);
      return a1;
    }
  }
  else
  {
    v17 = _mm_loadu_si128(v18);
    if ( (unsigned __int8)sub_233F860((char *)v17.m128i_i64[0], v17.m128i_u64[1], a2[108].m128i_i64) )
    {
      sub_2377250(
        (unsigned __int64 *)&v23,
        a2,
        a3,
        (__int64)v18,
        0xCCCCCCCCCCCCCCCDLL * (((char *)v19 - (char *)v18) >> 3));
      v6 = (unsigned __int64)v23 & 0xFFFFFFFFFFFFFFFELL;
      if ( ((unsigned __int64)v23 & 0xFFFFFFFFFFFFFFFELL) != 0 )
      {
        v23 = 0;
        *a1 = v6 | 1;
      }
      else
      {
        v23 = 0;
        sub_9C66B0((__int64 *)&v23);
        *a1 = 1;
      }
      sub_9C66B0((__int64 *)&v23);
    }
    else
    {
      v12 = sub_C63BB0();
      v24 = 45;
      v13 = v12;
      v15 = v14;
      v23 = "unknown function pass '{0}' in pipeline '{1}'";
      v25 = v31;
      v29 = v16;
      v26 = 2;
      v28 = &unk_49DB108;
      v30[0] = &unk_49DB108;
      v30[1] = &v17;
      v31[0] = v30;
      v31[1] = &v28;
      v27 = 1;
      sub_23328D0((__int64)v21, (__int64)&v23);
      sub_23058C0(a1, (__int64)v21, v13, v15);
      sub_2240A30(v21);
    }
  }
  if ( v20 )
    goto LABEL_11;
  return a1;
}
