// Function: sub_98CE00
// Address: 0x98ce00
//
__int64 __fastcall sub_98CE00(__int64 a1)
{
  __int64 v1; // r12
  __int64 v2; // rbx
  char *v3; // rdi
  __int64 result; // rax

  v1 = a1 + 48;
  v2 = *(_QWORD *)(a1 + 56);
  if ( v2 == a1 + 48 )
    return 1;
  while ( 1 )
  {
    v3 = (char *)(v2 - 24);
    if ( !v2 )
      v3 = 0;
    result = sub_98CD80(v3);
    if ( !(_BYTE)result )
      break;
    v2 = *(_QWORD *)(v2 + 8);
    if ( v1 == v2 )
      return 1;
  }
  return result;
}
