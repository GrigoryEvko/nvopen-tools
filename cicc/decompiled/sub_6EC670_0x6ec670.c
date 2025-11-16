// Function: sub_6EC670
// Address: 0x6ec670
//
__int64 __fastcall sub_6EC670(__int64 a1, __int64 a2, int a3, int a4)
{
  __int64 v5; // r12
  __int64 v6; // rax
  __int64 v7; // r15
  unsigned __int64 i; // rax
  __int64 v10; // rdx
  __int64 v11; // rax

  v5 = qword_4F04C68[0] + 776LL * (int)dword_4F04C5C;
  v6 = sub_726700(5);
  *(_QWORD *)(v6 + 56) = a2;
  v7 = v6;
  *(_BYTE *)(v6 + 25) = a3 & 1 | *(_BYTE *)(v6 + 25) & 0xFE;
  if ( a3 )
    *(_QWORD *)v6 = a1;
  else
    *(_QWORD *)v6 = sub_73D720(a1);
  if ( a4 )
  {
    *(_BYTE *)(a2 + 50) |= 0x10u;
    if ( word_4D04898 )
    {
      v11 = sub_730290(a2);
      if ( a2 != v11 )
        *(_BYTE *)(v11 + 50) |= 0x10u;
    }
  }
  for ( i = *(unsigned __int8 *)(v5 + 4); (_BYTE)i == 7; i = *(unsigned __int8 *)(v5 + 4) )
    v5 = qword_4F04C68[0] + 776LL * *(int *)(v5 + 552);
  if ( (unsigned __int8)i > 0xCu || (v10 = 4866, !_bittest64(&v10, i)) )
    sub_732EF0(v5);
  sub_6EB510(v7);
  if ( dword_4D041E8 && qword_4D03C50 && (unsigned int)sub_6EC5C0(a1, 0) )
    *(_BYTE *)(qword_4D03C50 + 20LL) |= 0x40u;
  return v7;
}
