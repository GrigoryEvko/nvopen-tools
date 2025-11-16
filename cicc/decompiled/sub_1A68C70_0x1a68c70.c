// Function: sub_1A68C70
// Address: 0x1a68c70
//
__int64 __fastcall sub_1A68C70(__int64 a1, char a2)
{
  __int64 result; // rax

  result = (unsigned __int8)byte_4FB4B80;
  *(_QWORD *)(a1 + 8) = 0;
  if ( !a2 )
    a2 = result;
  *(_BYTE *)a1 = a2;
  return result;
}
