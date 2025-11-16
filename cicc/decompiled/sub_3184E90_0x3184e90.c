// Function: sub_3184E90
// Address: 0x3184e90
//
__int64 __fastcall sub_3184E90(__int64 a1)
{
  __int64 v1; // rcx
  __int64 result; // rax

  v1 = *(_QWORD *)(a1 + 16);
  result = 0;
  if ( (unsigned __int8)(*(_BYTE *)v1 - 22) > 6u )
    return *(_DWORD *)(v1 + 4) & 0x7FFFFFF;
  return result;
}
