// Function: sub_1C28CE0
// Address: 0x1c28ce0
//
__int64 __fastcall sub_1C28CE0(__int64 a1, __int64 a2)
{
  __int64 *v4; // rdx
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v7; // r14
  _BYTE *v8; // rcx
  _BYTE *v9; // rax
  char i; // dl
  __int64 v11; // rdi
  __m128i *v12; // rdx
  __int64 v13; // rax
  _QWORD *v14; // rdx
  __int64 v15; // rdi
  __int64 v16; // rdi
  _BYTE *v17; // rax

  sub_1649960(a2);
  if ( !(unsigned __int8)sub_160E740() )
    return 0;
  v4 = *(__int64 **)(a1 + 8);
  v5 = *v4;
  v6 = v4[1];
  if ( v5 == v6 )
LABEL_20:
    BUG();
  while ( *(_UNKNOWN **)v5 != &unk_5054414 )
  {
    v5 += 16;
    if ( v6 == v5 )
      goto LABEL_20;
  }
  v7 = *(_QWORD *)((*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v5 + 8) + 104LL))(
                     *(_QWORD *)(v5 + 8),
                     &unk_5054414)
                 + 160);
  v8 = (_BYTE *)qword_4F9E580[21];
  v9 = (_BYTE *)qword_4F9E580[20];
  for ( i = 0; v8 != v9; ++v9 )
    i |= *v9;
  if ( (i & 1) != 0 )
  {
    v11 = *(_QWORD *)(a1 + 160);
    v12 = *(__m128i **)(v11 + 24);
    if ( *(_QWORD *)(v11 + 16) - (_QWORD)v12 <= 0xFu )
    {
      v11 = sub_16E7EE0(v11, "Max Live RRegs: ", 0x10u);
    }
    else
    {
      *v12 = _mm_load_si128((const __m128i *)&xmmword_3F6EF60);
      *(_QWORD *)(v11 + 24) += 16LL;
    }
    v13 = sub_16E7AB0(v11, *(int *)(v7 + 24));
    v14 = *(_QWORD **)(v13 + 24);
    v15 = v13;
    if ( *(_QWORD *)(v13 + 16) - (_QWORD)v14 <= 7u )
    {
      v15 = sub_16E7EE0(v13, "\tPRegs: ", 8u);
    }
    else
    {
      *v14 = 0x203A736765525009LL;
      *(_QWORD *)(v13 + 24) += 8LL;
    }
    v16 = sub_16E7AB0(v15, *(int *)(v7 + 28));
    v17 = *(_BYTE **)(v16 + 24);
    if ( *(_BYTE **)(v16 + 16) == v17 )
    {
      sub_16E7EE0(v16, "\t", 1u);
    }
    else
    {
      *v17 = 9;
      ++*(_QWORD *)(v16 + 24);
    }
  }
  sub_1C28B20((__int64 *)(a1 + 160), a2);
  sub_1C28930((__int64 *)(a1 + 160), *(char **)(a1 + 168), *(_QWORD *)(a1 + 176), a2);
  return 0;
}
