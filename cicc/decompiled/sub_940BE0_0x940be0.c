// Function: sub_940BE0
// Address: 0x940be0
//
__int64 __fastcall sub_940BE0(__int64 *a1, __int64 a2)
{
  const __m128i *v2; // r15
  __int64 v3; // rdx
  __int64 v4; // rcx
  unsigned __int64 v5; // r8
  int v6; // r9d
  const char *v7; // rax
  __int64 *i; // r13
  __int64 *v9; // rdx
  unsigned int v10; // eax
  __int64 v11; // rcx
  size_t v12; // rax
  __int64 v13; // rax
  const char *v14; // r14
  int v15; // ebx
  __int64 v16; // rcx
  __int64 v17; // rax
  __int64 v18; // r14
  __int64 v19; // r15
  __int64 v20; // rax
  char v21; // dl
  _BYTE *v22; // rax
  int v23; // r12d
  char v24; // bl
  int v25; // ecx
  __int64 v26; // rsi
  __int64 v27; // r12
  __int64 v29; // rax
  int v30; // [rsp+4h] [rbp-CCh]
  __int64 v31; // [rsp+8h] [rbp-C8h]
  char *s; // [rsp+18h] [rbp-B8h]
  __int64 v33; // [rsp+28h] [rbp-A8h]
  __int64 v34; // [rsp+30h] [rbp-A0h]
  int v35; // [rsp+44h] [rbp-8Ch] BYREF
  __int64 v36; // [rsp+48h] [rbp-88h]
  _BYTE *v37; // [rsp+50h] [rbp-80h] BYREF
  __int64 v38; // [rsp+58h] [rbp-78h]
  _BYTE v39[112]; // [rsp+60h] [rbp-70h] BYREF

  v2 = 0;
  if ( (**(_BYTE **)(a2 + 176) & 1) != 0 )
  {
    v2 = *(const __m128i **)(a2 + 168);
    if ( (*(_BYTE *)(a2 + 161) & 0x10) != 0 )
      v2 = (const __m128i *)v2[6].m128i_i64[0];
  }
  sub_93ED80(*(_DWORD *)(a2 + 64), (char *)&v35);
  v7 = byte_3F871B3;
  if ( *(_QWORD *)(a2 + 8) )
    v7 = *(const char **)(a2 + 8);
  s = (char *)v7;
  v31 = 8LL * *(_QWORD *)(a2 + 128);
  if ( *(char *)(a2 + 142) >= 0 && *(_BYTE *)(a2 + 140) == 12 )
    v30 = 8 * sub_8D4AB0(a2);
  else
    v30 = 8 * *(_DWORD *)(a2 + 136);
  v37 = v39;
  v38 = 0x800000000LL;
  for ( i = a1 + 2; v2; LODWORD(v38) = v38 + 1 )
  {
    v14 = (const char *)v2->m128i_i64[1];
    if ( !v14 )
      v14 = byte_3F871B3;
    v15 = sub_620E90((__int64)v2);
    v17 = sub_91FFA0(*a1, v2, 0, v16);
    if ( v15 )
    {
      v9 = *(__int64 **)(v17 + 24);
      v10 = *(_DWORD *)(v17 + 32);
      if ( v10 > 0x40 )
      {
        v11 = *v9;
      }
      else
      {
        v11 = 0;
        if ( v10 )
          v11 = (__int64)((_QWORD)v9 << (64 - (unsigned __int8)v10)) >> (64 - (unsigned __int8)v10);
      }
    }
    else
    {
      v11 = *(_QWORD *)(v17 + 24);
      if ( *(_DWORD *)(v17 + 32) > 0x40u )
        v11 = *(_QWORD *)v11;
    }
    v33 = v11;
    v12 = strlen(v14);
    v13 = sub_ADC8E0(i, v14, v12, v33, v15 == 0);
    v3 = (unsigned int)v38;
    v5 = (unsigned int)v38 + 1LL;
    if ( v5 > HIDWORD(v38) )
    {
      v34 = v13;
      sub_C8D5F0(&v37, v39, (unsigned int)v38 + 1LL, 8);
      v3 = (unsigned int)v38;
      v13 = v34;
    }
    v4 = (__int64)v37;
    *(_QWORD *)&v37[8 * v3] = v13;
    v2 = (const __m128i *)v2[7].m128i_i64[1];
  }
  v18 = sub_9405D0((__int64)a1, *(_DWORD *)(a2 + 64), v3, v4, v5, v6);
  v19 = sub_ADCD70(i, v37, (unsigned int)v38);
  if ( (*(_BYTE *)(a2 + 141) & 8) != 0 )
    goto LABEL_31;
  v20 = *(_QWORD *)(a2 + 40);
  v21 = *(_BYTE *)(a2 + 89) & 2;
  if ( v20 )
  {
    v22 = (_BYTE *)(v20 - 8);
    if ( !v21 )
      goto LABEL_26;
    goto LABEL_37;
  }
  if ( !v21 )
  {
LABEL_31:
    BYTE4(v36) = 0;
    v23 = v35;
    v25 = strlen(s);
    goto LABEL_32;
  }
LABEL_37:
  v22 = (_BYTE *)(sub_72F070(a2) - 8);
LABEL_26:
  v23 = v35;
  v24 = *v22 & 1;
  BYTE4(v36) = 0;
  v25 = strlen(s);
  if ( v24 )
  {
    v26 = v18;
    goto LABEL_28;
  }
LABEL_32:
  v26 = v18;
  v29 = a1[64];
  if ( v29 != a1[60] )
  {
    if ( v29 == a1[65] )
      v29 = *(_QWORD *)(a1[67] - 8) + 512LL;
    v26 = *(_QWORD *)(v29 - 8);
  }
LABEL_28:
  v27 = sub_ADEBE0((_DWORD)i, v26, (_DWORD)s, v25, v18, v23, v31, v30, v19, 0, 0, (__int64)byte_3F871B3, 0, 0, v36);
  if ( v37 != v39 )
    _libc_free(v37, v26);
  return v27;
}
