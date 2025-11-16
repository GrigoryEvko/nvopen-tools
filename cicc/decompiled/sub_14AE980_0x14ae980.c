// Function: sub_14AE980
// Address: 0x14ae980
//
__int64 __fastcall sub_14AE980(__int64 a1)
{
  __int64 v1; // r12
  __int64 v2; // rbx
  __int64 v3; // rdi
  __int64 result; // rax

  v1 = a1 + 40;
  v2 = *(_QWORD *)(a1 + 48);
  if ( v2 == a1 + 40 )
    return 1;
  while ( 1 )
  {
    v3 = v2 - 24;
    if ( !v2 )
      v3 = 0;
    result = sub_14AE440(v3);
    if ( !(_BYTE)result )
      break;
    v2 = *(_QWORD *)(v2 + 8);
    if ( v1 == v2 )
      return 1;
  }
  return result;
}
