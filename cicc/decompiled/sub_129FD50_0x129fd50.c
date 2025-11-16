// Function: sub_129FD50
// Address: 0x129fd50
//
__int64 __fastcall sub_129FD50(__int64 *a1, __int64 a2)
{
  const __m128i *v2; // r15
  const char *v3; // rax
  __int64 *i; // r13
  unsigned int v5; // edx
  __int64 *v6; // rax
  __int64 v7; // rcx
  size_t v8; // rax
  __int64 v9; // r8
  __int64 v10; // rax
  const char *v11; // r14
  int v12; // ebx
  __int64 v13; // rax
  __int64 v14; // r14
  __int64 v15; // r15
  __int64 v16; // rax
  char v17; // dl
  char *v18; // rax
  char v19; // bl
  int v20; // r12d
  int v21; // ecx
  __int64 v22; // rsi
  __int64 v23; // r12
  __int64 v25; // rax
  int v26; // [rsp+4h] [rbp-CCh]
  __int64 v27; // [rsp+8h] [rbp-C8h]
  char *s; // [rsp+18h] [rbp-B8h]
  __int64 v29; // [rsp+28h] [rbp-A8h]
  __int64 v30; // [rsp+30h] [rbp-A0h]
  int v31; // [rsp+4Ch] [rbp-84h] BYREF
  _BYTE *v32; // [rsp+50h] [rbp-80h] BYREF
  __int64 v33; // [rsp+58h] [rbp-78h]
  _BYTE v34[112]; // [rsp+60h] [rbp-70h] BYREF

  v2 = 0;
  if ( (**(_BYTE **)(a2 + 176) & 1) != 0 )
  {
    v2 = *(const __m128i **)(a2 + 168);
    if ( (*(_BYTE *)(a2 + 161) & 0x10) != 0 )
      v2 = (const __m128i *)v2[6].m128i_i64[0];
  }
  sub_129E300(*(_DWORD *)(a2 + 64), (char *)&v31);
  v3 = byte_3F871B3;
  if ( *(_QWORD *)(a2 + 8) )
    v3 = *(const char **)(a2 + 8);
  s = (char *)v3;
  v27 = 8LL * *(_QWORD *)(a2 + 128);
  if ( *(char *)(a2 + 142) >= 0 && *(_BYTE *)(a2 + 140) == 12 )
    v26 = 8 * sub_8D4AB0(a2);
  else
    v26 = 8 * *(_DWORD *)(a2 + 136);
  v32 = v34;
  v33 = 0x800000000LL;
  for ( i = a1 + 2; v2; LODWORD(v33) = v33 + 1 )
  {
    v11 = (const char *)v2->m128i_i64[1];
    if ( !v11 )
      v11 = byte_3F871B3;
    v12 = sub_620E90((__int64)v2);
    v13 = sub_127F610(*a1, v2, 0);
    if ( v12 )
    {
      v5 = *(_DWORD *)(v13 + 32);
      v6 = *(__int64 **)(v13 + 24);
      if ( v5 > 0x40 )
        v7 = *v6;
      else
        v7 = (__int64)((_QWORD)v6 << (64 - (unsigned __int8)v5)) >> (64 - (unsigned __int8)v5);
    }
    else
    {
      v7 = *(_QWORD *)(v13 + 24);
      if ( *(_DWORD *)(v13 + 32) > 0x40u )
        v7 = *(_QWORD *)v7;
    }
    v29 = v7;
    v8 = strlen(v11);
    v9 = sub_15A5960(i, v11, v8, v29, v12 == 0);
    v10 = (unsigned int)v33;
    if ( (unsigned int)v33 >= HIDWORD(v33) )
    {
      v30 = v9;
      sub_16CD150(&v32, v34, 0, 8);
      v10 = (unsigned int)v33;
      v9 = v30;
    }
    *(_QWORD *)&v32[8 * v10] = v9;
    v2 = (const __m128i *)v2[7].m128i_i64[1];
  }
  v14 = sub_129F850((__int64)a1, *(_DWORD *)(a2 + 64));
  v15 = sub_15A5DC0(i, v32, (unsigned int)v33);
  if ( (*(_BYTE *)(a2 + 141) & 8) != 0 )
    goto LABEL_30;
  v16 = *(_QWORD *)(a2 + 40);
  v17 = *(_BYTE *)(a2 + 89) & 2;
  if ( v16 )
  {
    v18 = (char *)(v16 - 8);
    if ( !v17 )
      goto LABEL_25;
    goto LABEL_36;
  }
  if ( !v17 )
  {
LABEL_30:
    v20 = v31;
    v21 = strlen(s);
    goto LABEL_31;
  }
LABEL_36:
  v18 = (char *)(sub_72F070(a2) - 8);
LABEL_25:
  v19 = *v18;
  v20 = v31;
  v21 = strlen(s);
  if ( (v19 & 1) != 0 )
  {
    v22 = v14;
    goto LABEL_27;
  }
LABEL_31:
  v22 = v14;
  v25 = a1[68];
  if ( v25 != a1[64] )
  {
    if ( v25 == a1[69] )
      v25 = *(_QWORD *)(a1[71] - 8) + 512LL;
    v22 = *(_QWORD *)(v25 - 8);
  }
LABEL_27:
  v23 = sub_15A6D60((_DWORD)i, v22, (_DWORD)s, v21, v14, v20, v27, v26, v15, 0, (__int64)byte_3F871B3, 0, 0);
  if ( v32 != v34 )
    _libc_free(v32, v22);
  return v23;
}
