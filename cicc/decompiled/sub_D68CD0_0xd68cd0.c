// Function: sub_D68CD0
// Address: 0xd68cd0
//
__int64 __fastcall sub_D68CD0(unsigned __int64 *a1, unsigned int a2, _QWORD *a3)
{
  __int64 result; // rax

  a1[1] = 0;
  *a1 = 2LL * a2;
  result = a3[2];
  a1[2] = result;
  if ( result != -4096 && result != 0 && result != -8192 )
    return sub_BD6050(a1, *a3 & 0xFFFFFFFFFFFFFFF8LL);
  return result;
}
