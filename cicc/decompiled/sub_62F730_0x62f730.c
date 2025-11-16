// Function: sub_62F730
// Address: 0x62f730
//
__int64 __fastcall sub_62F730(__int64 *a1, __int64 a2, int a3)
{
  __int64 i; // r12
  __int64 v5; // rbx
  __int64 result; // rax

  for ( i = *a1; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  v5 = sub_7259C0(8);
  sub_73C230(i, v5);
  if ( a3 )
  {
    *(_BYTE *)(v5 + 168) |= 0x80u;
    *(_QWORD *)(v5 + 176) = 0;
  }
  else
  {
    *(_QWORD *)(v5 + 176) = a2;
    if ( !a2 )
      *(_BYTE *)(v5 + 169) |= 0x20u;
  }
  result = sub_8D6090(v5);
  *a1 = v5;
  return result;
}
