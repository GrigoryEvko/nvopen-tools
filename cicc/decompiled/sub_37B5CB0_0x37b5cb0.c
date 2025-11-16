// Function: sub_37B5CB0
// Address: 0x37b5cb0
//
__int64 __fastcall sub_37B5CB0(__int64 *a1, unsigned int a2)
{
  __int64 v2; // rax
  __int64 (*v3)(void); // rdx
  __int64 (*v4)(); // rdx
  __int64 result; // rax

  v2 = *a1;
  v3 = *(__int64 (**)(void))(*a1 + 24);
  if ( v3 != sub_36FAC00 )
  {
    result = v3();
    if ( (_BYTE)result )
      return result;
    v2 = *a1;
  }
  v4 = *(__int64 (**)())(v2 + 16);
  result = 0;
  if ( v4 != sub_37B5C90 )
    return ((__int64 (__fastcall *)(__int64 *, _QWORD))v4)(a1, a2);
  return result;
}
