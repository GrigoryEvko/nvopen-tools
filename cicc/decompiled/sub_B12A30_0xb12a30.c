// Function: sub_B12A30
// Address: 0xb12a30
//
__int64 __fastcall sub_B12A30(__int64 a1)
{
  __int64 v1; // rdx
  __int64 result; // rax

  v1 = *(_QWORD *)(a1 + 40);
  result = 1;
  if ( *(_BYTE *)v1 == 4 )
    return *(unsigned int *)(v1 + 144);
  return result;
}
