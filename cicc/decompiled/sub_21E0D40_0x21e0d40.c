// Function: sub_21E0D40
// Address: 0x21e0d40
//
__int64 __fastcall sub_21E0D40(__int64 a1, __int64 a2)
{
  __int64 v3; // rsi
  __int64 v4; // r12
  _QWORD *v5; // r15
  __int64 v6; // r14
  __int64 v7; // r13
  const char *v8; // r14
  char *v9; // rax
  size_t v10; // r9
  char *v11; // rdx
  const char *v12; // r14
  __int64 *v13; // rax
  size_t v14; // r8
  int v15; // r9d
  __int64 *v16; // r13
  size_t v17; // rax
  _BYTE *v18; // r10
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 v21; // rcx
  int v22; // r8d
  __int64 v23; // rdx
  __int64 v24; // rdi
  const __m128i *v25; // rdx
  __m128i v26; // xmm0
  __int64 v27; // r9
  __int64 v28; // r12
  char *v30; // rax
  char *v31; // rdi
  __int64 v32; // rax
  size_t n; // [rsp+8h] [rbp-A8h]
  size_t na; // [rsp+8h] [rbp-A8h]
  _QWORD *v35; // [rsp+10h] [rbp-A0h]
  __int64 v36; // [rsp+20h] [rbp-90h] BYREF
  int v37; // [rsp+28h] [rbp-88h]
  char *s[2]; // [rsp+30h] [rbp-80h] BYREF
  _QWORD v39[2]; // [rsp+40h] [rbp-70h] BYREF
  __m128i v40; // [rsp+50h] [rbp-60h] BYREF
  __int64 v41; // [rsp+60h] [rbp-50h]
  __int64 v42; // [rsp+68h] [rbp-48h]
  __m128i v43; // [rsp+70h] [rbp-40h]

  v3 = *(_QWORD *)(a2 + 72);
  v4 = *(_QWORD *)(a1 + 16);
  v5 = *(_QWORD **)(a1 - 176);
  v36 = v3;
  if ( v3 )
    sub_1623A60((__int64)&v36, v3, 2);
  v37 = *(_DWORD *)(a2 + 64);
  v6 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a2 + 32) + 120LL) + 88LL);
  v7 = sub_3936750();
  sub_3936BD0(v6, v7);
  v8 = (const char *)sub_3936860(v7, 1);
  s[0] = (char *)v39;
  if ( !v8 )
    goto LABEL_29;
  v9 = (char *)strlen(v8);
  v40.m128i_i64[0] = (__int64)v9;
  v10 = (size_t)v9;
  if ( (unsigned __int64)v9 > 0xF )
  {
    n = (size_t)v9;
    v30 = (char *)sub_22409D0(s, &v40, 0);
    v10 = n;
    s[0] = v30;
    v31 = v30;
    v39[0] = v40.m128i_i64[0];
  }
  else
  {
    if ( v9 == (char *)1 )
    {
      LOBYTE(v39[0]) = *v8;
      v11 = (char *)v39;
      goto LABEL_7;
    }
    if ( !v9 )
    {
      v11 = (char *)v39;
      goto LABEL_7;
    }
    v31 = (char *)v39;
  }
  memcpy(v31, v8, v10);
  v9 = (char *)v40.m128i_i64[0];
  v11 = s[0];
LABEL_7:
  s[1] = v9;
  v9[(_QWORD)v11] = 0;
  sub_39367A0(v7);
  v12 = s[0];
  v13 = (__int64 *)sub_22077B0(32);
  v16 = v13;
  if ( v13 )
  {
    *v13 = (__int64)(v13 + 2);
    v35 = v13 + 2;
    if ( v12 )
    {
      v17 = strlen(v12);
      v18 = v35;
      v40.m128i_i64[0] = v17;
      v14 = v17;
      if ( v17 > 0xF )
      {
        na = v17;
        v32 = sub_22409D0(v16, &v40, 0);
        v14 = na;
        *v16 = v32;
        v18 = (_BYTE *)v32;
        v16[2] = v40.m128i_i64[0];
      }
      else
      {
        if ( v17 == 1 )
        {
          *((_BYTE *)v16 + 16) = *v12;
LABEL_12:
          v16[1] = v17;
          v18[v17] = 0;
          goto LABEL_13;
        }
        if ( !v17 )
          goto LABEL_12;
      }
      memcpy(v18, v12, v14);
      v17 = v40.m128i_i64[0];
      v18 = (_BYTE *)*v16;
      goto LABEL_12;
    }
LABEL_29:
    sub_426248((__int64)"basic_string::_M_construct null not valid");
  }
LABEL_13:
  v19 = *(unsigned int *)(v4 + 83296);
  if ( (unsigned int)v19 >= *(_DWORD *)(v4 + 83300) )
  {
    sub_16CD150(v4 + 83288, (const void *)(v4 + 83304), 0, 8, v14, v15);
    v19 = *(unsigned int *)(v4 + 83296);
  }
  *(_QWORD *)(*(_QWORD *)(v4 + 83288) + 8 * v19) = v16;
  ++*(_DWORD *)(v4 + 83296);
  v20 = sub_1D2F9D0(v5, (const char *)*v16, 5u, 0, 0);
  v21 = *(_QWORD *)(a2 + 40);
  v22 = *(_DWORD *)(a2 + 60);
  v24 = v23;
  v25 = *(const __m128i **)(a2 + 32);
  v26 = _mm_loadu_si128(v25 + 5);
  v41 = v20;
  v42 = v24;
  v40 = v26;
  v43 = _mm_loadu_si128(v25);
  v28 = sub_1D23DE0(v5, 198, (__int64)&v36, v21, v22, v27, v40.m128i_i64, 3);
  if ( (_QWORD *)s[0] != v39 )
    j_j___libc_free_0(s[0], v39[0] + 1LL);
  if ( v36 )
    sub_161E7C0((__int64)&v36, v36);
  return v28;
}
