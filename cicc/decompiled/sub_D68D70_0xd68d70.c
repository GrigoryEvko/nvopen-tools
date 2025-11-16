// Function: sub_D68D70
// Address: 0xd68d70
//
__int64 __fastcall sub_D68D70(_QWORD *a1)
{
  __int64 result; // rax

  result = a1[2];
  if ( result != -4096 && result != 0 && result != -8192 )
    return sub_BD60C0(a1);
  return result;
}
