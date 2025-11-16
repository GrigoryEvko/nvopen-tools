// Function: sub_3568140
// Address: 0x3568140
//
__int64 __fastcall sub_3568140(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 *v4; // rbx
  __int64 *v5; // r14
  __int64 v7; // r12
  __int64 result; // rax

  v4 = *(__int64 **)(a2 + 64);
  v5 = &v4[*(unsigned int *)(a2 + 72)];
  if ( v4 == v5 )
    return 1;
  while ( 1 )
  {
    v7 = *v4;
    if ( (unsigned __int8)sub_2E6D360(*(_QWORD *)(a1 + 8), a3, *v4) )
    {
      result = sub_2E6D360(*(_QWORD *)(a1 + 8), a4, v7);
      if ( !(_BYTE)result )
        break;
    }
    if ( v5 == ++v4 )
      return 1;
  }
  return result;
}
