// Function: sub_7DFBB0
// Address: 0x7dfbb0
//
unsigned __int64 __fastcall sub_7DFBB0(__int64 a1)
{
  __int64 v1; // rbx
  int i; // eax
  unsigned __int64 result; // rax

  v1 = *(_QWORD *)(a1 + 120);
  for ( i = *(unsigned __int8 *)(v1 + 140); (_BYTE)i == 12; i = *(unsigned __int8 *)(v1 + 140) )
    v1 = *(_QWORD *)(v1 + 160);
  result = (unsigned int)(i - 9);
  if ( (unsigned __int8)result <= 1u )
  {
    result = *(_QWORD *)(v1 + 168);
    if ( *(_QWORD *)(result + 32) < *(_QWORD *)(v1 + 128) )
    {
      result = sub_730E80(v1);
      if ( result < *(_QWORD *)(v1 + 128) )
        *(_BYTE *)(a1 + 145) |= 0x10u;
    }
  }
  return result;
}
