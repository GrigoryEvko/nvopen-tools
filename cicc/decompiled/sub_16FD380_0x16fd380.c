// Function: sub_16FD380
// Address: 0x16fd380
//
__int64 __fastcall sub_16FD380(__int64 a1, unsigned __int64 a2)
{
  __int64 result; // rax
  __int64 v3; // rdx
  __int64 v4; // rcx
  __int64 v5; // r8
  __int64 v6; // r13
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // rax
  unsigned int v11; // edx
  __m128i v12; // xmm0
  _BYTE *v13; // rsi
  __int64 v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // r8
  __int64 v17; // rdx
  const char *v18; // rax
  __int64 v19; // rdi
  __int64 *v20; // rax
  _QWORD *v21; // r14
  __int64 v22; // r9
  unsigned int v23; // [rsp+0h] [rbp-A0h] BYREF
  __m128i v24; // [rsp+8h] [rbp-98h]
  _QWORD *v25; // [rsp+18h] [rbp-88h] BYREF
  _QWORD v26[3]; // [rsp+28h] [rbp-78h] BYREF
  _QWORD v27[2]; // [rsp+40h] [rbp-60h] BYREF
  char v28; // [rsp+50h] [rbp-50h]
  char v29; // [rsp+51h] [rbp-4Fh]
  _QWORD *v30; // [rsp+58h] [rbp-48h]
  _QWORD v31[7]; // [rsp+68h] [rbp-38h] BYREF

  result = sub_16F91D0(a1);
  if ( (_BYTE)result )
    goto LABEL_16;
  v6 = *(_QWORD *)(a1 + 80);
  if ( v6 )
  {
    result = sub_16FD110(*(_QWORD *)(a1 + 80), a2, v3, v4, v5);
    if ( result )
    {
      (*(void (__fastcall **)(__int64))(*(_QWORD *)result + 8LL))(result);
      result = (__int64)sub_16FD200(v6, a2, v7, v8, v9);
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
  v10 = sub_16FC340(a1, a2, v3, v4, v5);
  v11 = *(_DWORD *)v10;
  v12 = _mm_loadu_si128((const __m128i *)(v10 + 8));
  v25 = v26;
  v13 = *(_BYTE **)(v10 + 24);
  v23 = v11;
  v14 = *(_QWORD *)(v10 + 32);
  v24 = v12;
  sub_16F6740((__int64 *)&v25, v13, (__int64)&v13[v14]);
  result = v23;
  if ( (v23 & 0xFFFFFFFD) != 0x10 )
  {
    v17 = *(unsigned int *)(a1 + 72);
    if ( (_DWORD)v17 )
    {
      if ( v23 == 11 )
      {
        sub_16FC240((__int64)v27, a1, v17, v15, v16);
        if ( v30 != v31 )
          j_j___libc_free_0(v30, v31[0] + 1LL);
        result = sub_16FD380(a1);
        v19 = (__int64)v25;
        if ( v25 != v26 )
          return j_j___libc_free_0(v19, v26[0] + 1LL);
        return result;
      }
      if ( v23 != 15 )
      {
        if ( !v23 )
          goto LABEL_13;
        v29 = 1;
        v18 = "Unexpected token. Expected Key, Flow Entry, or Flow Mapping End.";
        goto LABEL_12;
      }
    }
    else
    {
      if ( !v23 )
      {
LABEL_13:
        *(_BYTE *)(a1 + 77) = 1;
        *(_QWORD *)(a1 + 80) = 0;
        goto LABEL_14;
      }
      if ( v23 != 8 )
      {
        v29 = 1;
        v18 = "Unexpected token. Expected Key or Block End";
LABEL_12:
        v27[0] = v18;
        v28 = 3;
        result = (__int64)sub_16F8380(a1, (__int64)v27, (__int64)&v23, v15, v16);
        goto LABEL_13;
      }
    }
    sub_16FC240((__int64)v27, a1, v17, v15, v16);
    result = (__int64)v31;
    if ( v30 != v31 )
      result = j_j___libc_free_0(v30, v31[0] + 1LL);
    goto LABEL_13;
  }
  v20 = (__int64 *)sub_16F82D0(a1);
  v21 = (_QWORD *)sub_145CBF0(v20, 88, 16);
  sub_16FC350((__int64)v21, 3u, *(_QWORD *)(a1 + 8), 0, 0, v22, 0);
  *(_QWORD *)(a1 + 80) = v21;
  v21[9] = 0;
  result = (__int64)&unk_49EFE18;
  v21[10] = 0;
  *v21 = &unk_49EFE18;
LABEL_14:
  v19 = (__int64)v25;
  if ( v25 != v26 )
    return j_j___libc_free_0(v19, v26[0] + 1LL);
  return result;
}
