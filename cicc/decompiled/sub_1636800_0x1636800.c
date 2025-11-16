// Function: sub_1636800
// Address: 0x1636800
//
__int64 __fastcall sub_1636800(__int64 a1, __int64 *a2)
{
  __int64 v3; // rdi
  __int64 (*v4)(); // rcx
  __int64 result; // rax

  v3 = sub_16033D0(*a2);
  v4 = *(__int64 (**)())(*(_QWORD *)v3 + 16LL);
  result = 0;
  if ( v4 != sub_1635EC0 )
    return ((unsigned int (__fastcall *)(__int64, __int64, __int64 *))v4)(v3, a1, a2) ^ 1;
  return result;
}
