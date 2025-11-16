// Function: sub_6FE2E0
// Address: 0x6fe2e0
//
__int64 __fastcall sub_6FE2E0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  char v6; // al
  _DWORD *v7; // rcx
  __int64 v8; // r13
  __int64 v9; // rax
  __int64 v10; // rdx
  bool v11; // bl
  __int64 v12; // rax
  __int64 *v13; // rdi
  __int64 v15; // r14
  _QWORD *v16; // rbx
  __int64 v17; // r13
  int v18; // eax
  __int64 v19; // r15
  __int64 v20; // rax
  __int64 v21; // rax
  __int64 v22; // [rsp+8h] [rbp-58h]
  int v23; // [rsp+24h] [rbp-3Ch] BYREF
  __int64 v24; // [rsp+28h] [rbp-38h] BYREF

  v6 = *(_BYTE *)(a1 + 8);
  if ( !v6 )
  {
    v7 = &dword_4F04C44;
    v8 = *(_QWORD *)(a1 + 24);
    v9 = qword_4D03C50;
    v10 = *(unsigned __int8 *)(v8 + 24);
    v11 = (*(_BYTE *)(qword_4D03C50 + 19LL) & 2) != 0;
    if ( (dword_4F04C44 != -1
       || (v7 = (_DWORD *)(qword_4F04C68[0] + 776LL * dword_4F04C64), (*((_BYTE *)v7 + 6) & 6) != 0)
       || *((_BYTE *)v7 + 4) == 12)
      && (_BYTE)v10 == 3 )
    {
      sub_6F3BA0((__m128i *)(v8 + 8), 0);
    }
    else
    {
      if ( (_BYTE)v10 == 5 )
        goto LABEL_9;
      sub_6F6C80((_QWORD *)(v8 + 8));
    }
    if ( *(_BYTE *)(v8 + 24) == 2 && *(_BYTE *)(v8 + 325) == 12 && *(_BYTE *)(v8 + 328) == 1 )
    {
      v21 = sub_72E9A0(v8 + 152);
      v13 = (__int64 *)v21;
      if ( v21 )
      {
        if ( *(_QWORD *)(v21 + 80) )
          return sub_6F8A60(v13, a2, v10, (__int64)v7, a5, a6);
      }
    }
    v9 = qword_4D03C50;
LABEL_9:
    *(_BYTE *)(v9 + 19) |= 2u;
    v12 = sub_6F6F40((const __m128i *)(v8 + 8), 0, v10, (__int64)v7, a5, a6);
    v7 = (_DWORD *)qword_4D03C50;
    v13 = (__int64 *)v12;
    v10 = *(_BYTE *)(qword_4D03C50 + 19LL) & 0xFD;
    *(_BYTE *)(qword_4D03C50 + 19LL) = *(_BYTE *)(qword_4D03C50 + 19LL) & 0xFD | (2 * v11);
    return sub_6F8A60(v13, a2, v10, (__int64)v7, a5, a6);
  }
  if ( v6 != 1 )
    sub_721090(a1);
  v22 = sub_6E2F40(1);
  *(_BYTE *)(v22 + 9) = *(_BYTE *)(a1 + 9) & 4 | *(_BYTE *)(v22 + 9) & 0xFB;
  *(__m128i *)(v22 + 24) = _mm_loadu_si128((const __m128i *)(a1 + 24));
  *(_QWORD *)(v22 + 40) = *(_QWORD *)(a1 + 40);
  v15 = *(_QWORD *)(a1 + 24);
  if ( v15 )
  {
    v16 = 0;
    v17 = 0;
    while ( 1 )
    {
      v18 = sub_869530(
              *(_QWORD *)(v15 + 16),
              *(_QWORD *)(a2 + 32),
              *(_QWORD *)(a2 + 24),
              (unsigned int)&v24,
              *(_DWORD *)(a2 + 40),
              *(_QWORD *)(a2 + 48),
              (__int64)&v23);
      if ( v23 )
        *(_BYTE *)(a2 + 56) = 1;
      if ( v18 )
        break;
      v19 = (__int64)v16;
LABEL_25:
      if ( (*(_BYTE *)(a2 + 40) & 0x40) == 0 || !sub_6E1B40(v15) )
        goto LABEL_26;
      if ( !v17 )
      {
        v19 = v15;
        v17 = v15;
LABEL_26:
        v20 = *(_QWORD *)v15;
        if ( !*(_QWORD *)v15 )
          goto LABEL_32;
        goto LABEL_27;
      }
      *(_QWORD *)v19 = v15;
      v20 = *(_QWORD *)v15;
      v19 = v15;
      if ( !*(_QWORD *)v15 )
        goto LABEL_32;
LABEL_27:
      if ( *(_BYTE *)(v20 + 8) == 3 )
      {
        v20 = sub_6BBB10((_QWORD *)v15);
        if ( !v20 )
          goto LABEL_32;
      }
      v16 = (_QWORD *)v19;
      v15 = v20;
    }
    while ( 1 )
    {
      if ( *(_BYTE *)(v15 + 8) == 2 )
      {
        v19 = sub_6E2F40(2);
        *(__m128i *)(v19 + 24) = _mm_loadu_si128((const __m128i *)(v15 + 24));
        *(__m128i *)(v19 + 40) = _mm_loadu_si128((const __m128i *)(v15 + 40));
        *(_QWORD *)(v19 + 56) = *(_QWORD *)(v15 + 56);
        if ( v17 )
        {
LABEL_19:
          *v16 = v19;
          goto LABEL_20;
        }
      }
      else
      {
        v19 = sub_6FE2E0(v15, a2);
        if ( v17 )
          goto LABEL_19;
      }
      v17 = v19;
LABEL_20:
      sub_867630(v24, 0);
      if ( !(unsigned int)sub_866C00(v24) )
        goto LABEL_25;
      v16 = (_QWORD *)v19;
    }
  }
  v17 = 0;
LABEL_32:
  *(_QWORD *)(v22 + 24) = v17;
  return v22;
}
