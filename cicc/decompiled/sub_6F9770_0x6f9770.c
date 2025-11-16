// Function: sub_6F9770
// Address: 0x6f9770
//
__int64 __fastcall sub_6F9770(__int64 a1, unsigned int a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rax
  __int64 i; // rdx

  if ( !*(_BYTE *)(a1 + 16) )
    return sub_6E6870(a1);
  v6 = *(_QWORD *)a1;
  for ( i = *(unsigned __int8 *)(*(_QWORD *)a1 + 140LL); (_BYTE)i == 12; i = *(unsigned __int8 *)(v6 + 140) )
    v6 = *(_QWORD *)(v6 + 160);
  if ( (_BYTE)i )
    return sub_6F9470((const __m128i *)a1, a2, i, a4, a5, a6);
  else
    return sub_6E6870(a1);
}
