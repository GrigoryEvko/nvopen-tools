// Function: sub_317ED00
// Address: 0x317ed00
//
_QWORD *__fastcall sub_317ED00(__int64 a1, __int64 a2, int *a3, size_t a4)
{
  __int64 v6; // r12
  __int64 v8[5]; // [rsp+8h] [rbp-28h] BYREF

  v6 = sub_317E6A0(a1, a2);
  if ( v6 )
  {
    v8[0] = sub_C1B090(a2, 0);
    return sub_317E540(v6, v8, a3, a4);
  }
  return (_QWORD *)v6;
}
