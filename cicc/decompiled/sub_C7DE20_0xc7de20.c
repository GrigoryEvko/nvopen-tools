// Function: sub_C7DE20
// Address: 0xc7de20
//
_QWORD *__fastcall sub_C7DE20(_QWORD *a1, const void *a2, unsigned __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // [rsp+0h] [rbp-30h] BYREF
  char v8; // [rsp+10h] [rbp-20h]

  sub_C7DD80((__int64)&v7, a2, a3, a4, a5, a6);
  if ( (v8 & 1) != 0 )
    *a1 = 0;
  else
    *a1 = v7;
  return a1;
}
