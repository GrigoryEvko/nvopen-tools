// Function: sub_1691920
// Address: 0x1691920
//
__int64 __fastcall sub_1691920(_QWORD *a1, int a2)
{
  __int64 v3; // [rsp+0h] [rbp-10h] BYREF

  if ( a2 )
    sub_1690400(&v3, *a1 + ((unsigned __int64)(unsigned int)(a2 - 1) << 6), (__int64)a1);
  else
    sub_1690400(&v3, 0, 0);
  return v3;
}
