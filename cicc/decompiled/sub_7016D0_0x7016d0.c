// Function: sub_7016D0
// Address: 0x7016d0
//
__int64 __fastcall sub_7016D0(__int64 a1, __m128i *a2, __m128i *a3, __m128i *a4, __int64 a5, __int64 a6)
{
  __int64 v8; // r12
  __int64 v9; // rax
  unsigned int v10; // r15d
  __int64 v11; // rdx
  unsigned int v12; // edi
  char v13; // al
  __int64 v14; // rdx
  __int64 *v16; // r14
  __int64 v17; // rax
  __int64 v18; // rdi
  __int64 v19; // rdi
  __int64 *v20; // r13
  __int64 v21; // rdx
  __int64 v22; // rcx
  __int64 v23; // r8
  __int64 v24; // rax
  __int64 v25; // rdi
  __int64 v26; // rax
  int v27; // r13d
  __int64 v28; // rdx
  __int64 v29; // rcx
  __int64 v30; // r8
  __int64 v31; // r9
  __int64 v32; // rdx
  __int64 v33; // rcx
  __int64 v34; // r8
  __int64 v35; // r9
  unsigned int v36; // [rsp+4h] [rbp-4Ch]
  __int64 v38; // [rsp+10h] [rbp-40h] BYREF
  __int64 v39[7]; // [rsp+18h] [rbp-38h] BYREF

  v8 = a2->m128i_i64[0];
  if ( !a2->m128i_i64[0] )
  {
    v16 = (__int64 *)a2;
    if ( (unsigned int)sub_8DBE70(*(_QWORD *)a1) )
      return 0;
    a2 = a4;
    v17 = sub_7955B0(a1, a4);
    v8 = v17;
    if ( !v17 )
      return 0;
    *v16 = v17;
  }
  if ( (*(_BYTE *)(v8 + 193) & 4) != 0 )
  {
    v36 = 1;
    *(_BYTE *)(a1 + 60) |= 4u;
    v9 = qword_4D03C50;
  }
  else
  {
    if ( !(unsigned int)sub_6DEB30(a1, a2) )
      return 0;
    v9 = qword_4D03C50;
    if ( (*(_BYTE *)(qword_4D03C50 + 18LL) & 0x40) != 0 )
      return 0;
    v36 = 0;
  }
  if ( *(char *)(v9 + 19) < 0 )
  {
    a2 = (__m128i *)dword_4D048AC;
    if ( !dword_4D048AC )
      return 0;
  }
  v38 = sub_724DC0(a1, a2, a3, a4, a5, a6);
  v10 = sub_7A39E0(a1, v36, v38, a4);
  if ( v10 )
  {
    v18 = v38;
    if ( (*(_BYTE *)(qword_4D03C50 + 21LL) & 0x40) != 0 && a1 && !*(_QWORD *)(v38 + 144) )
    {
      sub_731DB0(a1);
      v18 = v38;
    }
    sub_6E6A50(v18, (__int64)a3);
    v19 = a3->m128i_i64[0];
    *(__int64 *)((char *)a3[4].m128i_i64 + 4) = *(_QWORD *)(a1 + 28);
    if ( (unsigned int)sub_8D2FB0(v19) )
    {
      v27 = sub_8D3110(a3->m128i_i64[0]);
      sub_6F82C0((__int64)a3, (__int64)a3, v28, v29, v30, v31);
      if ( v27 )
        sub_6ED1A0((__int64)a3);
    }
    else if ( *(_BYTE *)(qword_4D03C50 + 16LL) > 3u && (unsigned int)sub_8D3A70(a3->m128i_i64[0]) )
    {
      v20 = (__int64 *)sub_6ECAE0(a3->m128i_i64[0], 0, 0, 1, 2u, (__int64 *)(a1 + 28), v39);
      v24 = sub_724E50(&v38, 0, v21, v22, v23);
      sub_72F900(v39[0], v24);
      sub_6E70E0(v20, (__int64)a3);
      goto LABEL_9;
    }
  }
  else if ( v36 )
  {
    if ( (unsigned int)sub_731EE0(a1) )
    {
      sub_6E70E0((__int64 *)a1, (__int64)a3);
      sub_6F4B70(a3, (__int64)a3, v32, v33, v34, v35);
    }
    else if ( qword_4D03C50 && (*(_BYTE *)(qword_4D03C50 + 24LL) & 3) != 0 )
    {
      v25 = sub_726700(36);
      *(_QWORD *)v25 = *(_QWORD *)a1;
      v26 = *(_QWORD *)(a1 + 28);
      *(_QWORD *)(v25 + 56) = a1;
      *(_QWORD *)(v25 + 28) = v26;
      *(_DWORD *)(v25 + 64) = *(_DWORD *)(qword_4D03C50 + 24LL);
      sub_6E70E0((__int64 *)v25, (__int64)a3);
    }
    else if ( (unsigned int)sub_6E5270(v8, v38, (_DWORD *)(a1 + 28), a4) )
    {
      sub_6E6A50(v38, (__int64)a3);
      v10 = v36;
    }
  }
  sub_724E30(&v38);
LABEL_9:
  if ( ((*(_BYTE *)(v8 + 193) & 8) != 0 && *(_DWORD *)(v8 + 224) == 1
     || !*(_BYTE *)(v8 + 174) && *(_WORD *)(v8 + 176) == 10203)
    && (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 12) & 0x10) == 0
    && qword_4F04C50 )
  {
    v11 = *(_QWORD *)(qword_4F04C50 + 32LL);
    if ( (*(_DWORD *)(v11 + 192) & 0x8000400) == 0x400 )
    {
      v12 = 3059;
    }
    else
    {
      if ( (*(_BYTE *)(v11 + 193) & 2) != 0 )
        return v10;
      if ( *(_BYTE *)(qword_4D03C50 + 16LL) <= 3u )
        return v10;
      if ( *(_DWORD *)(unk_4D03B98 + 176LL * unk_4D03B90) == 2 )
        return v10;
      v12 = 3060;
      if ( (*(_BYTE *)(qword_4D03C50 + 18LL) & 2) != 0 )
        return v10;
    }
    v13 = *(_BYTE *)(v8 + 89);
    v14 = 0;
    if ( (v13 & 0x40) == 0 )
    {
      if ( (v13 & 8) != 0 )
        v14 = *(_QWORD *)(v8 + 24);
      else
        v14 = *(_QWORD *)(v8 + 8);
    }
    sub_684B10(v12, (_DWORD *)(a1 + 28), v14);
  }
  return v10;
}
