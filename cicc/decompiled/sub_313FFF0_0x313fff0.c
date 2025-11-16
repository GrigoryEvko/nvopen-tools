// Function: sub_313FFF0
// Address: 0x313fff0
//
__int64 __fastcall sub_313FFF0(__int64 *a1, __int64 a2)
{
  __m128i v2; // rax
  __int64 v3; // rbx
  __int64 v4; // rax
  __int64 v5; // rbx
  const void *v6; // r12
  size_t v7; // r13
  _BYTE *v8; // r15
  _BYTE *v9; // rbx
  _BYTE *v10; // r14
  const void *v12; // r14
  size_t v13; // rdx
  _BYTE *v14; // [rsp+10h] [rbp-E0h]
  _BYTE *v15; // [rsp+18h] [rbp-D8h]
  __m128i v16; // [rsp+20h] [rbp-D0h] BYREF
  _BYTE *v17; // [rsp+30h] [rbp-C0h] BYREF
  __int64 v18; // [rsp+38h] [rbp-B8h]
  _BYTE v19[176]; // [rsp+40h] [rbp-B0h] BYREF

  v17 = v19;
  v18 = 0x800000000LL;
  v2.m128i_i64[0] = sub_A72240(a1);
  v16 = v2;
  sub_C937F0(&v16, (__int64)&v17, ",", 1u, 0xFFFFFFFFLL, 1);
  v3 = 16LL * (unsigned int)v18;
  v14 = v17;
  v15 = &v17[v3];
  v4 = v3 >> 4;
  v5 = v3 >> 6;
  if ( v5 )
  {
    v6 = *(const void **)a2;
    v7 = *(_QWORD *)(a2 + 8);
    v8 = v17;
    v9 = &v17[64 * v5];
    while ( *((_QWORD *)v8 + 1) != v7 || v7 && memcmp(*(const void **)v8, v6, v7) )
    {
      if ( v7 == *((_QWORD *)v8 + 3) && ((v10 = v8 + 16, !v7) || !memcmp(*((const void **)v8 + 2), v6, v7))
        || v7 == *((_QWORD *)v8 + 5) && ((v10 = v8 + 32, !v7) || !memcmp(*((const void **)v8 + 4), v6, v7))
        || v7 == *((_QWORD *)v8 + 7) && ((v10 = v8 + 48, !v7) || !memcmp(*((const void **)v8 + 6), v6, v7)) )
      {
        LOBYTE(v6) = v15 != v10;
        goto LABEL_17;
      }
      v8 += 64;
      if ( v9 == v8 )
      {
        v4 = (v15 - v8) >> 4;
        goto LABEL_22;
      }
    }
    goto LABEL_16;
  }
  v8 = v17;
LABEL_22:
  switch ( v4 )
  {
    case 2LL:
      v12 = *(const void **)a2;
      v6 = *(const void **)(a2 + 8);
      break;
    case 3LL:
      v12 = *(const void **)a2;
      v6 = *(const void **)(a2 + 8);
      if ( v6 == *((const void **)v8 + 1)
        && (!v6 || !memcmp(*(const void **)v8, *(const void **)a2, *(_QWORD *)(a2 + 8))) )
      {
        goto LABEL_16;
      }
      v8 += 16;
      break;
    case 1LL:
      v12 = *(const void **)a2;
      v6 = *(const void **)(a2 + 8);
      goto LABEL_29;
    default:
LABEL_25:
      LODWORD(v6) = 0;
      goto LABEL_17;
  }
  if ( v6 == *((const void **)v8 + 1) && (!v6 || !memcmp(*(const void **)v8, v12, (size_t)v6)) )
    goto LABEL_16;
  v8 += 16;
LABEL_29:
  if ( v6 != *((const void **)v8 + 1) )
    goto LABEL_25;
  if ( !v6 || (v13 = (size_t)v6, LODWORD(v6) = 0, !memcmp(*(const void **)v8, v12, v13)) )
LABEL_16:
    LOBYTE(v6) = v15 != v8;
LABEL_17:
  if ( v14 != v19 )
    _libc_free((unsigned __int64)v14);
  return (unsigned int)v6;
}
