// Function: sub_1992730
// Address: 0x1992730
//
__int64 **__fastcall sub_1992730(__int64 **a1, __int64 **a2, __int64 a3, __int64 a4)
{
  __int64 **i; // r12
  __int64 *v7; // r13
  __int64 v8; // rax

  for ( i = a1; a2 != i; i += 3 )
  {
    v7 = *i;
    if ( *((_BYTE *)*i + 16) > 0x17u && sub_1456C80(a4, *v7) )
    {
      v8 = sub_146F1B0(a4, (__int64)v7);
      if ( *(_WORD *)(v8 + 24) == 7 && a3 == *(_QWORD *)(v8 + 48) )
        break;
    }
  }
  return i;
}
