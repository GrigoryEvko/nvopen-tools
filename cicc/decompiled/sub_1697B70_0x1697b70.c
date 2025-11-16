// Function: sub_1697B70
// Address: 0x1697b70
//
unsigned __int64 *__fastcall sub_1697B70(unsigned __int64 *a1, __int64 a2, __int64 a3, char a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r12
  __int64 v7; // r15
  _BYTE **v8; // rcx
  size_t v9; // r14
  unsigned __int64 v10; // rax
  size_t v11; // r14
  const void *v12; // rbx
  _QWORD *v14; // rdi
  const __m128i **v15; // [rsp+0h] [rbp-160h]
  _BYTE *v17; // [rsp+10h] [rbp-150h]
  __int64 v18; // [rsp+20h] [rbp-140h]
  __int64 v20; // [rsp+38h] [rbp-128h] BYREF
  _QWORD v21[2]; // [rsp+40h] [rbp-120h] BYREF
  _BYTE *v22; // [rsp+50h] [rbp-110h] BYREF
  size_t v23; // [rsp+58h] [rbp-108h]
  _QWORD v24[2]; // [rsp+60h] [rbp-100h] BYREF
  _QWORD *v25; // [rsp+70h] [rbp-F0h] BYREF
  size_t v26; // [rsp+78h] [rbp-E8h]
  _QWORD v27[2]; // [rsp+80h] [rbp-E0h] BYREF
  __int64 v28[26]; // [rsp+90h] [rbp-D0h] BYREF

  v6 = a2;
  v7 = *(_QWORD *)(a3 + 32);
  v8 = &v22;
  v18 = a3 + 24;
  if ( a3 + 24 == v7 )
  {
LABEL_26:
    *(_BYTE *)(v6 + 128) = 0;
    sub_16977B0(v6, a2, a3, (__int64)v8, a5, a6);
    *a1 = 1;
    return a1;
  }
  while ( 1 )
  {
    if ( !v7 )
      BUG();
    if ( (*(_BYTE *)(v7 - 33) & 0x20) == 0 )
      goto LABEL_25;
    sub_1695660((__int64)&v22, v7 - 56, a4);
    sub_1696B90(v28, v6, v22, v23);
    if ( (v28[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
    {
      *a1 = v28[0] & 0xFFFFFFFFFFFFFFFELL | 1;
      goto LABEL_6;
    }
    v9 = v23;
    v15 = (const __m128i **)(v6 + 80);
    v17 = v22;
    v21[0] = v7 - 56;
    sub_16C1840(v28);
    sub_16C1A90(v28, v17, v9);
    sub_16C1AA0(v28, &v25);
    a2 = *(_QWORD *)(v6 + 88);
    v28[0] = (__int64)v25;
    if ( a2 == *(_QWORD *)(v6 + 96) )
    {
      sub_1695D20(v15, (const __m128i *)a2, v28, v21);
    }
    else
    {
      if ( a2 )
      {
        *(_QWORD *)a2 = v25;
        *(_QWORD *)(a2 + 8) = v21[0];
        a2 = *(_QWORD *)(v6 + 88);
      }
      a2 += 16;
      *(_QWORD *)(v6 + 88) = a2;
    }
    if ( a4 )
    {
      a2 = 46;
      v10 = sub_22417D0(&v22, 46, 0);
      if ( v10 != -1 )
        break;
    }
LABEL_23:
    if ( v22 != (_BYTE *)v24 )
    {
      a2 = v24[0] + 1LL;
      j_j___libc_free_0(v22, v24[0] + 1LL);
    }
LABEL_25:
    v7 = *(_QWORD *)(v7 + 8);
    if ( v18 == v7 )
      goto LABEL_26;
  }
  v25 = v27;
  if ( v10 > v23 )
    v10 = v23;
  sub_1693C00((__int64 *)&v25, v22, (__int64)&v22[v10]);
  sub_1696B90(v28, v6, v25, v26);
  if ( (v28[0] & 0xFFFFFFFFFFFFFFFELL) == 0 )
  {
    v20 = v7 - 56;
    v11 = v26;
    v12 = v25;
    sub_16C1840(v28);
    sub_16C1A90(v28, v12, v11);
    sub_16C1AA0(v28, v21);
    a2 = *(_QWORD *)(v6 + 88);
    v28[0] = v21[0];
    if ( a2 == *(_QWORD *)(v6 + 96) )
    {
      sub_1695D20(v15, (const __m128i *)a2, v28, &v20);
    }
    else
    {
      if ( a2 )
      {
        *(_QWORD *)a2 = v21[0];
        *(_QWORD *)(a2 + 8) = v20;
        a2 = *(_QWORD *)(v6 + 88);
      }
      a2 += 16;
      *(_QWORD *)(v6 + 88) = a2;
    }
    if ( v25 != v27 )
    {
      a2 = v27[0] + 1LL;
      j_j___libc_free_0(v25, v27[0] + 1LL);
    }
    goto LABEL_23;
  }
  v14 = v25;
  *a1 = v28[0] & 0xFFFFFFFFFFFFFFFELL | 1;
  if ( v14 != v27 )
    j_j___libc_free_0(v14, v27[0] + 1LL);
LABEL_6:
  if ( v22 != (_BYTE *)v24 )
    j_j___libc_free_0(v22, v24[0] + 1LL);
  return a1;
}
