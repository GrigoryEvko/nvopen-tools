// Function: sub_660F90
// Address: 0x660f90
//
__int64 __fastcall sub_660F90(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  int v6; // r13d
  __int64 v7; // rbx
  __int64 v8; // rax
  __int64 v9; // rcx
  _DWORD *v10; // rdx
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // r8
  __m128i *v14; // rax
  bool v15; // zf
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // r8
  unsigned __int8 v19; // r14
  __int64 v20; // rdx
  __int64 v21; // rcx
  __int64 v22; // rdx
  __int64 v23; // r14
  __int64 result; // rax
  __m128i **v25; // rdi
  __int64 v26; // r13
  __int64 v27; // rdx
  __int64 v28; // rcx
  __int64 v29; // r8
  __int64 v30; // rax
  __int64 v31; // rdx
  __int64 v32; // rcx
  __int64 v33; // rdx
  __int64 v34; // rcx
  __int64 v35; // [rsp-10h] [rbp-240h]
  __m128i **v36; // [rsp-8h] [rbp-238h]
  __m128i *v37; // [rsp+8h] [rbp-228h] BYREF
  _QWORD v38[2]; // [rsp+10h] [rbp-220h] BYREF
  _QWORD v39[66]; // [rsp+20h] [rbp-210h] BYREF

  v6 = 0;
  v7 = a1;
  v8 = sub_724DC0(a1, a2, a3, a4, a5, a6);
  v10 = &dword_4F04C5C;
  v37 = (__m128i *)v8;
  if ( dword_4F04C5C != dword_4F04C34 )
  {
    a2 = (__int64)dword_4F07508;
    a1 = 335;
    v6 = 1;
    sub_6851C0(335, dword_4F07508);
  }
  v38[0] = *(_QWORD *)(v7 + 24);
  sub_7B8B50(a1, a2, v10, v9);
  v14 = v37;
  *v37 = _mm_loadu_si128(xmmword_4F06300);
  v14[1] = _mm_loadu_si128(&xmmword_4F06300[1]);
  v14[2] = _mm_loadu_si128(&xmmword_4F06300[2]);
  v14[3] = _mm_loadu_si128(&xmmword_4F06300[3]);
  v14[4] = _mm_loadu_si128(&xmmword_4F06300[4]);
  v14[5] = _mm_loadu_si128(&xmmword_4F06300[5]);
  v14[6] = _mm_loadu_si128(&xmmword_4F06300[6]);
  v14[7] = _mm_loadu_si128(&xmmword_4F06300[7]);
  v14[8] = _mm_loadu_si128(xmmword_4F06380);
  v14[9] = _mm_loadu_si128(&xmmword_4F06380[1]);
  v14[10] = _mm_loadu_si128((const __m128i *)&unk_4F063A0);
  v15 = unk_4F063AD == 0;
  v14[11] = _mm_loadu_si128((const __m128i *)&unk_4F063B0);
  v14[12] = _mm_loadu_si128(&xmmword_4F063C0);
  v38[1] = qword_4F063F0;
  if ( v15 )
    goto LABEL_7;
  a2 = unk_4F061E8 + 416LL;
  if ( (unsigned int)sub_73A2C0(xmmword_4F06300, unk_4F061E8 + 416LL, v11, v12, v13) )
  {
    v19 = 2;
  }
  else
  {
    a2 = unk_4F061E8 + 624LL;
    if ( !(unsigned int)sub_73A2C0(xmmword_4F06300, unk_4F061E8 + 624LL, v16, v17, v18) )
    {
      a2 = (__int64)dword_4F07508;
      sub_6851C0(336, dword_4F07508);
LABEL_7:
      v19 = (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 9) >> 1) & 7;
      goto LABEL_8;
    }
    v19 = 3;
  }
  if ( v6 )
    goto LABEL_7;
LABEL_8:
  sub_85F8B0(v19);
  sub_7B8B50(v19, a2, v20, v21);
  if ( word_4F06418[0] == 73 )
  {
    v39[0] = *(_QWORD *)&dword_4F063F8;
    v25 = &v37;
    v26 = sub_727210();
    *(_QWORD *)v26 = sub_724E50(&v37, a2, v27, v28, v29);
    v30 = v38[0];
    *(_BYTE *)(v26 + 8) = *(_BYTE *)(v26 + 8) & 0xF8 | v19 & 7;
    *(_QWORD *)(v26 + 12) = v30;
    if ( !dword_4F04C3C )
    {
      a2 = 70;
      v25 = (__m128i **)v26;
      sub_8699D0(v26, 70, 0);
    }
    sub_854AB0();
    sub_7B8B50(v25, a2, v31, v32);
    ++*(_BYTE *)(qword_4F061C8 + 82LL);
    if ( word_4F06418[0] != 9 && word_4F06418[0] != 74 )
    {
      do
      {
        sub_660E20((*(_BYTE *)(v7 + 125) & 0x40) != 0, *(_BYTE *)(v7 + 125) >> 7, 0, 0, 0, 0, 0);
        a2 = v35;
        v25 = v36;
      }
      while ( word_4F06418[0] != 74 && word_4F06418[0] != 9 );
    }
    sub_85F950(v25, a2);
    sub_869D70(v26, 70);
    *(_QWORD *)(v26 + 20) = *(_QWORD *)&dword_4F063F8;
    --*(_BYTE *)(qword_4F061C8 + 82LL);
    if ( word_4F06418[0] == 74 )
    {
      if ( (*(_BYTE *)(v7 + 126) & 1) != 0 )
        unk_4F04D84 = 1;
      result = sub_7B8B50(v26, 70, v33, v34);
      unk_4F04D84 = 0;
    }
    else
    {
      return sub_7BE200(67, 3196, v39);
    }
  }
  else
  {
    if ( word_4F06418[0] == 9 )
    {
      sub_6851C0(169, dword_4F07508);
      sub_85F950(169, dword_4F07508);
    }
    else
    {
      v22 = qword_4F04C68[0] + 776LL * dword_4F04C64;
      v23 = *(_QWORD *)(v22 + 624);
      memset(v39, 0, 0x1D8u);
      v39[19] = v39;
      v39[3] = *(_QWORD *)&dword_4F063F8;
      if ( dword_4F077BC )
      {
        if ( qword_4F077A8 <= 0x9F5Fu )
          BYTE2(v39[22]) |= 1u;
      }
      *(_QWORD *)(v22 + 624) = v7;
      sub_660E20(
        (*(_BYTE *)(v7 + 125) & 0x40) != 0,
        *(_BYTE *)(v7 + 125) >> 7,
        *(_BYTE *)(v7 + 126) & 1,
        0,
        0,
        (__int64)v38,
        v39);
      *(_QWORD *)v7 = v39[0];
      *(_QWORD *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 624) = v23;
    }
    return sub_724E30(&v37);
  }
  return result;
}
