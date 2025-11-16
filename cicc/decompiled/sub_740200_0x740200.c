// Function: sub_740200
// Address: 0x740200
//
__m128i *__fastcall sub_740200(__int64 a1)
{
  _DWORD *v1; // rdx
  _QWORD *v2; // rcx
  __int64 v3; // r8
  __int64 v4; // r9
  __int64 *v6; // r13
  __int64 v7; // rsi
  __int64 v8; // r12
  __int64 v9; // r15
  char v10; // al
  __int64 i; // rdi
  __int64 *v12; // r12
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // r8
  __int64 v16; // r9
  _QWORD *v17; // rax
  __int64 v18; // r14
  int v19; // r8d
  char v20; // al
  __m128i *v21; // [rsp+8h] [rbp-68h]
  char v22; // [rsp+8h] [rbp-68h]
  char v23; // [rsp+1Bh] [rbp-55h] BYREF
  int v24; // [rsp+1Ch] [rbp-54h] BYREF
  __int64 *v25; // [rsp+20h] [rbp-50h] BYREF
  _QWORD *v26; // [rsp+28h] [rbp-48h] BYREF
  __int64 v27[8]; // [rsp+30h] [rbp-40h] BYREF

  sub_72F9F0(a1, 0, &v23, &v25);
  if ( v23 == 1 )
    return (__m128i *)*v25;
  if ( v23 != 2 )
  {
    if ( v23 == 6 )
      return *(__m128i **)(*v25 + 56);
    return 0;
  }
  v6 = v25;
  v7 = word_4D04898;
  v8 = *v25;
  if ( word_4D04898 )
  {
    if ( *(_BYTE *)(v8 + 48) == 2 )
      return *(__m128i **)(v8 + 56);
    v9 = *(_QWORD *)(a1 + 120);
    if ( !(unsigned int)sub_8DBE70(v9) )
    {
      v17 = sub_724DC0();
      v18 = *(_QWORD *)(v8 + 8);
      v24 = 0;
      v26 = v17;
      v19 = (int)v17;
      v22 = 0;
      if ( v18 )
      {
        v20 = *(_BYTE *)(v18 + 177);
        *(_BYTE *)(v18 + 177) = 0;
        v22 = v20;
      }
      if ( (*(_BYTE *)(v8 - 8) & 1) != 0 && dword_4F07270[0] != unk_4F073B8 )
      {
        sub_7296C0(&v24);
        v19 = (int)v26;
      }
      v7 = (__int64)&dword_4F063F8;
      v27[0] = 0;
      v27[1] = 0;
      if ( (unsigned int)sub_7A1C60(v8, (unsigned int)&dword_4F063F8, v9, 1, v19, (unsigned int)v27, 0)
        && (*((_BYTE *)v26 + 173) != 6 || *((_BYTE *)v26 + 176) != 3)
        && (unsigned int)sub_71ACC0((__int64)v26) )
      {
        *v6 = (__int64)sub_725A70(2u);
        v7 = sub_724E50((__int64 *)&v26, &dword_4F063F8);
        sub_72F900(*v6, (_BYTE *)v7);
        *(_QWORD *)(*v6 + 8) = *(_QWORD *)(v8 + 8);
      }
      sub_67E3D0(v27);
      if ( v26 )
        sub_724E30((__int64)&v26);
      if ( v24 )
        sub_729730(v24);
      if ( v18 )
        *(_BYTE *)(v18 + 177) = v22;
    }
    v8 = *v25;
  }
  v10 = *(_BYTE *)(v8 + 48);
  if ( v10 == 2 )
    return *(__m128i **)(v8 + 56);
  if ( (*(_BYTE *)(a1 + 170) & 0x10) == 0 )
  {
    v1 = &dword_4F04C44;
    if ( dword_4F04C44 == -1 )
    {
      v2 = qword_4F04C68;
      v1 = (_DWORD *)(qword_4F04C68[0] + 776LL * dword_4F04C64);
      if ( (*((_BYTE *)v1 + 6) & 6) == 0 && *((_BYTE *)v1 + 4) != 12 )
        return 0;
    }
  }
  if ( v10 != 3 )
  {
    if ( v10 != 5 || !(unsigned int)sub_8DD3B0(*(_QWORD *)(a1 + 120)) )
      return 0;
    for ( i = *(_QWORD *)(a1 + 120); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
      ;
    v12 = (__int64 *)sub_6EC670(i, *v25, 0, 1);
LABEL_24:
    if ( v12 )
    {
      v27[0] = (__int64)sub_724DC0();
      sub_70FD90(v12, v27[0]);
      v21 = sub_7401F0(v27[0]);
      sub_724E30((__int64)v27);
      return v21;
    }
    return 0;
  }
  v12 = *(__int64 **)(v8 + 56);
  if ( !(unsigned int)sub_731EE0((__int64)v12, v7, (__int64)v1, (__int64)v2, v3, v4)
    || (unsigned int)sub_731D60((__int64)v12, v7, v13, v14, v15, v16) )
  {
    return 0;
  }
  if ( !*(_QWORD *)(*v25 + 40) )
    goto LABEL_24;
  return 0;
}
