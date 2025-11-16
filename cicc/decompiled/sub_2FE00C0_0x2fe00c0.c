// Function: sub_2FE00C0
// Address: 0x2fe00c0
//
__int64 __fastcall sub_2FE00C0(_QWORD *a1, __int64 a2, __int64 a3)
{
  __int64 v5; // r14
  __int64 v6; // r8
  int v7; // edx
  char v8; // al
  __int64 v10; // rcx
  void (__fastcall *v11)(_QWORD *, _QWORD, __int64, __int64, __int64, _QWORD, _QWORD, __int64, __int64); // r15
  __int64 v12; // rsi
  __int64 v13; // rax
  unsigned __int64 v14; // r8
  __int64 v15; // rdx
  unsigned __int64 v16; // r14
  __int64 v17; // r15
  const __m128i *v18; // r13
  const __m128i *i; // r15
  unsigned int v20; // edx
  __int64 v21; // rax
  unsigned __int8 v22; // al
  unsigned __int8 v23; // al
  unsigned __int64 v24; // rdx
  unsigned int v25; // [rsp+Ch] [rbp-34h]
  unsigned int v26; // [rsp+Ch] [rbp-34h]

  if ( (unsigned __int8)sub_2E8B940(a2) )
    return sub_2E88D70(a2, (unsigned __int16 *)(a1[1] - 280LL));
  v5 = *(_QWORD *)(a2 + 32);
  v6 = *(unsigned int *)(v5 + 8);
  v7 = *(_DWORD *)(v5 + 48);
  v8 = *(_BYTE *)(v5 + 44) & 1;
  if ( (_DWORD)v6 == v7 )
  {
    if ( v8 || (*(_DWORD *)(a2 + 40) & 0xFFFFFFu) > 2 )
      return sub_2E88D70(a2, (unsigned __int16 *)(a1[1] - 280LL));
  }
  else
  {
    if ( v8 )
      return sub_2E88D70(a2, (unsigned __int16 *)(a1[1] - 280LL));
    v10 = 0;
    v11 = *(void (__fastcall **)(_QWORD *, _QWORD, __int64, __int64, __int64, _QWORD, _QWORD, __int64, __int64))(*a1 + 496LL);
    if ( (unsigned int)(v7 - 1) <= 0x3FFFFFFE )
    {
      v23 = sub_2EAB300(v5 + 40);
      v6 = *(unsigned int *)(v5 + 8);
      v10 = v23;
    }
    v12 = 0;
    if ( (unsigned int)(v6 - 1) <= 0x3FFFFFFE )
    {
      v26 = v10;
      v22 = sub_2EAB300(v5);
      v6 = *(unsigned int *)(v5 + 8);
      v10 = v26;
      v12 = v22;
    }
    v11(
      a1,
      *(_QWORD *)(a2 + 24),
      a2,
      a2 + 56,
      v6,
      *(unsigned int *)(v5 + 48),
      ((*(_BYTE *)(v5 + 43) >> 4) ^ 1) & (*(_BYTE *)(v5 + 43) >> 6) & 1,
      v12,
      v10);
    v13 = *(_DWORD *)(a2 + 40) & 0xFFFFFF;
    if ( (unsigned int)v13 > 2 )
    {
      v14 = *(_QWORD *)a2 & 0xFFFFFFFFFFFFFFF8LL;
      if ( !v14 )
        BUG();
      v15 = *(_QWORD *)v14;
      v16 = *(_QWORD *)a2 & 0xFFFFFFFFFFFFFFF8LL;
      if ( (*(_QWORD *)v14 & 4) == 0 && (*(_BYTE *)(v14 + 44) & 4) != 0 )
      {
        while ( 1 )
        {
          v24 = v15 & 0xFFFFFFFFFFFFFFF8LL;
          v16 = v24;
          if ( (*(_BYTE *)(v24 + 44) & 4) == 0 )
            break;
          v15 = *(_QWORD *)v24;
        }
      }
      v17 = *(_QWORD *)(a2 + 32);
      v18 = (const __m128i *)(v17 + 40 * v13);
      v25 = *(_DWORD *)(v17 + 8);
      for ( i = (const __m128i *)(v17 + 40LL * (unsigned int)sub_2E88F80(a2));
            v18 != i;
            i = (const __m128i *)((char *)i + 40) )
      {
        sub_2E8F270(v16, i);
        if ( (((i->m128i_i8[3] & 0x40) != 0) & (((unsigned __int8)i->m128i_i8[3] >> 4) ^ 1)) != 0 )
        {
          v20 = i->m128i_u32[2];
          if ( v25 == v20 || v25 - 1 <= 0x3FFFFFFE && v20 - 1 <= 0x3FFFFFFE && (unsigned __int8)sub_E92070(a3, v25, v20) )
          {
            v21 = *(_QWORD *)(v16 + 32) + 40LL * ((*(_DWORD *)(v16 + 40) & 0xFFFFFFu) - 1);
            *(_BYTE *)(v21 + 3) &= ~0x40u;
          }
        }
      }
    }
  }
  return sub_2E88E20(a2);
}
