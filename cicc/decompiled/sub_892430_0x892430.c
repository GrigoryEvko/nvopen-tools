// Function: sub_892430
// Address: 0x892430
//
__m128i *__fastcall sub_892430(__int64 a1, __int64 a2)
{
  __int64 v3; // r15
  __int64 v4; // rbx
  __int64 v5; // r12
  unsigned int *v6; // rsi
  unsigned __int64 v7; // rdi
  __int64 v8; // r12
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // r8
  __int64 v12; // r9
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // r8
  __int64 *v16; // r9
  __int64 *v17; // r13
  int v18; // r15d
  int v19; // r14d
  int v20; // eax
  __m128i *result; // rax
  __int64 v23; // [rsp+10h] [rbp-220h]
  _BOOL4 v24; // [rsp+18h] [rbp-218h]
  __int64 *v25; // [rsp+18h] [rbp-218h]
  _QWORD v26[66]; // [rsp+20h] [rbp-210h] BYREF

  switch ( *(_BYTE *)(a2 + 80) )
  {
    case 4:
    case 5:
      v23 = *(_QWORD *)(*(_QWORD *)(a2 + 96) + 80LL);
      break;
    case 6:
      v23 = *(_QWORD *)(*(_QWORD *)(a2 + 96) + 32LL);
      break;
    case 9:
    case 0xA:
      v23 = *(_QWORD *)(*(_QWORD *)(a2 + 96) + 56LL);
      break;
    case 0x13:
    case 0x14:
    case 0x15:
    case 0x16:
      v23 = *(_QWORD *)(a2 + 88);
      break;
    default:
      BUG();
  }
  v3 = *(_QWORD *)(v23 + 176);
  v4 = *(_QWORD *)(v3 + 88);
  v5 = sub_892400(v23);
  v24 = sub_864700(*(_QWORD *)(v23 + 32), 0, 0, v3, a2, **(_QWORD **)(v4 + 168), 1, 0x802u);
  sub_854C10(*(const __m128i **)(v23 + 56));
  sub_7BC160(v5);
  *(_QWORD *)(*(_QWORD *)(v4 + 168) + 52LL) = *(_QWORD *)&dword_4F063F8;
  *(_BYTE *)(a2 + 83) |= 0x40u;
  memset(v26, 0, 0x1D8u);
  v26[19] = v26;
  v26[3] = *(_QWORD *)&dword_4F063F8;
  if ( dword_4F077BC && qword_4F077A8 <= 0x9F5Fu )
    BYTE2(v26[22]) |= 1u;
  BYTE3(v26[16]) |= 0x20u;
  BYTE2(v26[15]) |= 0x80u;
  sub_65C7C0((__int64)v26);
  sub_64EC60((__int64)v26);
  v6 = 0;
  v7 = v3;
  v8 = v26[36];
  *(_BYTE *)(a2 + 83) &= ~0x40u;
  *(_QWORD *)(*(_QWORD *)(v4 + 168) + 60LL) = *(_QWORD *)&dword_4F061D8;
  *(_QWORD *)(v4 + 160) = v8;
  sub_854980(v3, 0);
  if ( word_4F06418[0] != 9 )
  {
    v6 = &dword_4F063F8;
    v7 = 65;
    sub_6851C0(0x41u, &dword_4F063F8);
    while ( word_4F06418[0] != 9 )
      sub_7B8B50(0x41u, &dword_4F063F8, v9, v10, v11, v12);
  }
  sub_7B8B50(v7, v6, v9, v10, v11, v12);
  if ( v24 )
    sub_863FE0(v7, v24, v13, v14, v15, v16);
  *(_BYTE *)(v23 + 265) |= 2u;
  v25 = **(__int64 ***)(a1 + 192);
  if ( v25 )
  {
    v17 = **(__int64 ***)(a1 + 192);
    v18 = 0;
    v19 = 0;
    do
    {
      v20 = sub_88FAD0(*(_BYTE *)(v17[1] + 80), *(_QWORD *)(v17[1] + 88), v8, 0);
      *((_BYTE *)v17 + 57) = (4 * (v20 & 1)) | *((_BYTE *)v17 + 57) & 0xFB;
      v17 = (__int64 *)*v17;
      if ( v20 )
        v19 = 1;
      else
        v18 = 1;
    }
    while ( v17 );
    if ( !v19 && (unsigned int)sub_8D97B0(v8) )
    {
      do
      {
        *((_BYTE *)v25 + 57) |= 4u;
        v25 = (__int64 *)*v25;
      }
      while ( v25 );
    }
    if ( v18 )
      *(_BYTE *)(v23 + 267) |= 4u;
  }
  else
  {
    sub_8D97B0(v8);
  }
  result = (__m128i *)&dword_4F07590;
  if ( dword_4F07590 )
  {
    result = (__m128i *)a1;
    if ( *(_DWORD *)(a1 + 112) )
    {
      result = (__m128i *)dword_4F04C3C;
      if ( !dword_4F04C3C )
      {
        result = sub_8921F0(*(_QWORD *)(a1 + 336));
        *(__int16 *)((char *)&result[3].m128i_i16[4] + 1) |= 0x201u;
        result[2].m128i_i64[0] = v8;
      }
    }
  }
  return result;
}
