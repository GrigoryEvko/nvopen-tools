// Function: sub_CE1EA0
// Address: 0xce1ea0
//
__int64 __fastcall sub_CE1EA0(__int64 a1, __int64 a2)
{
  char *v3; // rax
  __int64 v4; // rdx
  __int64 *v6; // rdx
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // r14
  _BYTE *v10; // rax
  char v11; // dl
  __int64 v12; // rdi
  __m128i *v13; // rdx
  __int64 v14; // rax
  _QWORD *v15; // rdx
  __int64 v16; // rdi
  __int64 v17; // rdi
  _BYTE *v18; // rax

  v3 = (char *)sub_BD5D20(a2);
  if ( !sub_BC63A0(v3, v4) )
    return 0;
  v6 = *(__int64 **)(a1 + 8);
  v7 = *v6;
  v8 = v6[1];
  if ( v7 == v8 )
LABEL_20:
    BUG();
  while ( *(_UNKNOWN **)v7 != &unk_4F8D474 )
  {
    v7 += 16;
    if ( v8 == v7 )
      goto LABEL_20;
  }
  v9 = *(_QWORD *)((*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v7 + 8) + 104LL))(
                     *(_QWORD *)(v7 + 8),
                     &unk_4F8D474)
                 + 176);
  v10 = (_BYTE *)unk_4F83008;
  v11 = 0;
  if ( unk_4F83008 != unk_4F83010 )
  {
    do
      v11 |= *v10++;
    while ( (_BYTE *)unk_4F83010 != v10 );
  }
  if ( (v11 & 1) != 0 )
  {
    v12 = *(_QWORD *)(a1 + 176);
    v13 = *(__m128i **)(v12 + 32);
    if ( *(_QWORD *)(v12 + 24) - (_QWORD)v13 <= 0xFu )
    {
      v12 = sub_CB6200(v12, "Max Live RRegs: ", 0x10u);
    }
    else
    {
      *v13 = _mm_load_si128((const __m128i *)&xmmword_3F6EF60);
      *(_QWORD *)(v12 + 32) += 16LL;
    }
    v14 = sub_CB59F0(v12, *(int *)(v9 + 24));
    v15 = *(_QWORD **)(v14 + 32);
    v16 = v14;
    if ( *(_QWORD *)(v14 + 24) - (_QWORD)v15 <= 7u )
    {
      v16 = sub_CB6200(v14, "\tPRegs: ", 8u);
    }
    else
    {
      *v15 = 0x203A736765525009LL;
      *(_QWORD *)(v14 + 32) += 8LL;
    }
    v17 = sub_CB59F0(v16, *(int *)(v9 + 28));
    v18 = *(_BYTE **)(v17 + 32);
    if ( *(_BYTE **)(v17 + 24) == v18 )
    {
      sub_CB6200(v17, (unsigned __int8 *)"\t", 1u);
    }
    else
    {
      *v18 = 9;
      ++*(_QWORD *)(v17 + 32);
    }
  }
  sub_CE1CE0((__int64 *)(a1 + 176), a2);
  sub_CE19D0((__int64 *)(a1 + 176), *(unsigned __int8 **)(a1 + 184), *(_QWORD *)(a1 + 192), a2);
  return 0;
}
