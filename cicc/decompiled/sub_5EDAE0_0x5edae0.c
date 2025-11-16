// Function: sub_5EDAE0
// Address: 0x5edae0
//
__int64 __fastcall sub_5EDAE0(__int64 a1, __int64 a2, __int64 a3, _QWORD *a4)
{
  __int64 result; // rax

  result = sub_5E9D70(a1, a2, a3, 1, a4);
  if ( !result )
    return sub_5E9D70(a1, a2, a3, 0, a4);
  return result;
}
