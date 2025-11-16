// Function: sub_EB0CE0
// Address: 0xeb0ce0
//
__int64 __fastcall sub_EB0CE0(__int64 a1, __int64 a2)
{
  __int64 v2; // r13
  __int64 v3; // rax
  unsigned int v4; // r12d
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 v9; // r13
  _BYTE *v10; // rdi
  size_t v11; // r14
  void *v12; // rsi
  int v13; // [rsp+0h] [rbp-80h]
  __m128i v14; // [rsp+8h] [rbp-78h]
  __int64 v15; // [rsp+18h] [rbp-68h] BYREF
  unsigned int v16; // [rsp+20h] [rbp-60h]
  const char *v17; // [rsp+30h] [rbp-50h] BYREF
  char v18; // [rsp+50h] [rbp-30h]
  char v19; // [rsp+51h] [rbp-2Fh]

  v2 = a2;
  v3 = sub_ECD7B0(a1);
  v13 = *(_DWORD *)v3;
  v14 = _mm_loadu_si128((const __m128i *)(v3 + 8));
  v16 = *(_DWORD *)(v3 + 32);
  if ( v16 > 0x40 )
  {
    a2 = v3 + 24;
    sub_C43780((__int64)&v15, (const void **)(v3 + 24));
  }
  else
  {
    v15 = *(_QWORD *)(v3 + 24);
  }
  sub_EABFE0(a1);
  if ( v13 != 3 || *(_BYTE *)v14.m128i_i64[0] != 34 )
  {
    v19 = 1;
    v17 = "expected double quoted string after .print";
    v18 = 3;
    v4 = sub_ECDA70(a1, v2, &v17, 0, 0);
    goto LABEL_6;
  }
  v4 = sub_ECE000(a1);
  if ( (_BYTE)v4 )
    goto LABEL_6;
  v9 = (__int64)sub_CB7210(a1, a2, v6, v7, v8);
  if ( v14.m128i_i64[1] >= 2uLL )
  {
    v10 = *(_BYTE **)(v9 + 32);
    v11 = v14.m128i_i64[1] - 2;
    v12 = (void *)(v14.m128i_i64[0] + 1);
    if ( *(_QWORD *)(v9 + 24) - (_QWORD)v10 >= (unsigned __int64)(v14.m128i_i64[1] - 2) )
    {
      if ( v14.m128i_i64[1] != 2 )
      {
        memcpy(v10, v12, v11);
        v10 = (_BYTE *)(v11 + *(_QWORD *)(v9 + 32));
        *(_QWORD *)(v9 + 32) = v10;
      }
      goto LABEL_16;
    }
    v9 = sub_CB6200(v9, (unsigned __int8 *)v12, v11);
  }
  v10 = *(_BYTE **)(v9 + 32);
LABEL_16:
  if ( *(_QWORD *)(v9 + 24) <= (unsigned __int64)v10 )
  {
    sub_CB5D20(v9, 10);
  }
  else
  {
    *(_QWORD *)(v9 + 32) = v10 + 1;
    *v10 = 10;
  }
LABEL_6:
  if ( v16 > 0x40 && v15 )
    j_j___libc_free_0_0(v15);
  return v4;
}
