// Function: sub_CAEB90
// Address: 0xcaeb90
//
__int64 __fastcall sub_CAEB90(__int64 a1, unsigned __int64 a2)
{
  __int64 result; // rax
  __int64 v4; // rdx
  __int64 v5; // rcx
  __int64 v6; // r8
  __int64 v7; // r13
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // rax
  unsigned int v12; // edx
  __m128i v13; // xmm0
  _BYTE *v14; // rsi
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // r8
  __int64 v18; // rdx
  const char *v19; // rax
  __int64 v20; // rdi
  _QWORD *v21; // rax
  __int64 v22; // r9
  _QWORD *v23; // rdi
  __int64 v24; // rax
  _QWORD *v25; // r13
  unsigned int v26; // [rsp+0h] [rbp-A0h] BYREF
  __m128i v27; // [rsp+8h] [rbp-98h]
  _QWORD *v28; // [rsp+18h] [rbp-88h] BYREF
  _QWORD v29[3]; // [rsp+28h] [rbp-78h] BYREF
  _QWORD v30[3]; // [rsp+40h] [rbp-60h] BYREF
  _QWORD *v31; // [rsp+58h] [rbp-48h]
  char v32; // [rsp+60h] [rbp-40h]
  char v33; // [rsp+61h] [rbp-3Fh]
  _QWORD v34[7]; // [rsp+68h] [rbp-38h] BYREF

  result = sub_CA94D0(a1);
  if ( (_BYTE)result )
    goto LABEL_16;
  v7 = *(_QWORD *)(a1 + 80);
  if ( v7 )
  {
    result = sub_CAE820(*(_QWORD *)(a1 + 80), a2, v4, v5, v6);
    if ( result )
    {
      (*(void (__fastcall **)(__int64))(*(_QWORD *)result + 8LL))(result);
      result = (__int64)sub_CAE940(v7, a2, v8, v9, v10);
      if ( result )
        result = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)result + 8LL))(result);
    }
    if ( *(_DWORD *)(a1 + 72) == 2 )
    {
LABEL_16:
      *(_BYTE *)(a1 + 77) = 1;
      *(_QWORD *)(a1 + 80) = 0;
      return result;
    }
  }
  v11 = sub_CAD7B0(a1, a2, v4, v5, v6);
  v12 = *(_DWORD *)v11;
  v13 = _mm_loadu_si128((const __m128i *)(v11 + 8));
  v28 = v29;
  v14 = *(_BYTE **)(v11 + 24);
  v26 = v12;
  v15 = *(_QWORD *)(v11 + 32);
  v27 = v13;
  sub_CA64F0((__int64 *)&v28, v14, (__int64)&v14[v15]);
  result = v26;
  if ( (v26 & 0xFFFFFFFD) != 0x10 )
  {
    v18 = *(unsigned int *)(a1 + 72);
    if ( (_DWORD)v18 )
    {
      if ( v26 == 11 )
      {
        sub_CAD6B0((__int64)v30, a1, v18, v16, v17);
        if ( v31 != v34 )
          j_j___libc_free_0(v31, v34[0] + 1LL);
        result = sub_CAEB90(a1);
        v20 = (__int64)v28;
        if ( v28 != v29 )
          return j_j___libc_free_0(v20, v29[0] + 1LL);
        return result;
      }
      if ( v26 != 15 )
      {
        if ( !v26 )
          goto LABEL_13;
        v33 = 1;
        v19 = "Unexpected token. Expected Key, Flow Entry, or Flow Mapping End.";
        goto LABEL_12;
      }
    }
    else
    {
      if ( !v26 )
      {
LABEL_13:
        *(_BYTE *)(a1 + 77) = 1;
        *(_QWORD *)(a1 + 80) = 0;
        goto LABEL_14;
      }
      if ( v26 != 8 )
      {
        v33 = 1;
        v19 = "Unexpected token. Expected Key or Block End";
LABEL_12:
        v30[0] = v19;
        v32 = 3;
        result = (__int64)sub_CA8D00(a1, (__int64)v30, (__int64)&v26, v16, v17);
        goto LABEL_13;
      }
    }
    sub_CAD6B0((__int64)v30, a1, v18, v16, v17);
    result = (__int64)v34;
    if ( v31 != v34 )
      result = j_j___libc_free_0(v31, v34[0] + 1LL);
    goto LABEL_13;
  }
  v21 = (_QWORD *)sub_CA8A30(a1);
  v21[10] += 88LL;
  v23 = v21;
  v24 = *v21;
  v25 = (_QWORD *)((v24 + 15) & 0xFFFFFFFFFFFFFFF0LL);
  if ( v23[1] >= (unsigned __int64)(v25 + 11) && v24 )
    *v23 = v25 + 11;
  else
    v25 = (_QWORD *)sub_9D1E70((__int64)v23, 88, 88, 4);
  sub_CAD7C0((__int64)v25, 3u, *(_QWORD *)(a1 + 8), 0, 0, v22, 0);
  v25[9] = 0;
  v25[10] = 0;
  result = (__int64)&unk_49DCCD8;
  *v25 = &unk_49DCCD8;
  *(_QWORD *)(a1 + 80) = v25;
LABEL_14:
  v20 = (__int64)v28;
  if ( v28 != v29 )
    return j_j___libc_free_0(v20, v29[0] + 1LL);
  return result;
}
