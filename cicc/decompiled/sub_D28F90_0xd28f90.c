// Function: sub_D28F90
// Address: 0xd28f90
//
unsigned __int64 __fastcall sub_D28F90(__int64 *a1, __int64 a2, unsigned __int64 *a3)
{
  __int64 v4; // rdx
  unsigned __int64 result; // rax

  v4 = *a1;
  a1[10] += 112;
  result = (v4 + 7) & 0xFFFFFFFFFFFFFFF8LL;
  if ( a1[1] >= result + 112 && v4 )
    *a1 = result + 112;
  else
    result = sub_9D1E70((__int64)a1, 112, 112, 3);
  *a3 = result;
  *(_QWORD *)result = a1;
  *(_QWORD *)(result + 8) = a2;
  *(_QWORD *)(result + 16) = 0;
  *(_BYTE *)(result + 104) = 0;
  return result;
}
