// Function: sub_7360F0
// Address: 0x7360f0
//
__int64 __fastcall sub_7360F0(__int64 a1)
{
  _BYTE *v1; // r12
  __int64 result; // rax
  size_t v3; // rdx
  __m128i **v4; // r12
  __m128i *v5; // rax
  __int64 v6; // rcx
  char *v7; // rdx
  _QWORD *v8; // rax
  __int64 v9[3]; // [rsp+8h] [rbp-18h] BYREF

  v1 = sub_735B90(-1, a1, v9);
  result = v9[0];
  if ( *(_QWORD *)(v9[0] + 48) != a1 )
  {
    if ( qword_4F07A68 )
    {
      if ( qword_4F07A48 == 100 )
        sub_7302F0();
    }
    else
    {
      qword_4F07A68 = (void *)sub_823970(2400);
      v8 = (_QWORD *)sub_823970(800);
      qword_4F07A50 = (__int64)v8;
      *v8 = 0;
      v8[99] = 0;
      memset(
        (void *)((unsigned __int64)(v8 + 1) & 0xFFFFFFFFFFFFFFF8LL),
        0,
        8LL * (((unsigned int)v8 - (((_DWORD)v8 + 8) & 0xFFFFFFF8) + 800) >> 3));
    }
    if ( (_BYTE *)qword_4F07A60 != v1 )
    {
      if ( qword_4F07A60 )
        sub_7302F0();
      qword_4F07A60 = (__int64)v1;
      qword_4F07A58 = v9[0];
    }
    v3 = qword_4F07A48;
    v4 = (__m128i **)(qword_4F07A50 + 8 * qword_4F07A48);
    v5 = *v4;
    if ( !*v4 )
    {
      v5 = sub_725FD0();
      *v4 = v5;
      v5->m128i_i64[1] = (__int64)off_4B6D4E0;
      v3 = qword_4F07A48;
    }
    v5[7].m128i_i64[0] = 0;
    v5[9].m128i_i64[1] = *(_QWORD *)(a1 + 152);
    v6 = v9[0];
    *(_QWORD *)(*(_QWORD *)(v9[0] + 48) + 112LL) = v5;
    *(_QWORD *)(v6 + 48) = v5;
    v7 = (char *)qword_4F07A68 + 24 * v3;
    *(_QWORD *)v7 = a1;
    *((_QWORD *)v7 + 1) = v5;
    result = *(unsigned int *)(a1 + 64);
    ++qword_4F07A48;
    *((_DWORD *)v7 + 4) = result;
  }
  return result;
}
