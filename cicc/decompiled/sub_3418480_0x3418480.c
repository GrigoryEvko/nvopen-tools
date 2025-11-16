// Function: sub_3418480
// Address: 0x3418480
//
void __fastcall sub_3418480(unsigned __int16 *a1, __int64 a2)
{
  unsigned __int8 *v2[2]; // [rsp+0h] [rbp-30h] BYREF
  __int64 v3; // [rsp+10h] [rbp-20h] BYREF

  sub_3009980((__int64)v2, a1);
  sub_CB6200(a2, v2[0], (size_t)v2[1]);
  if ( (__int64 *)v2[0] != &v3 )
    j_j___libc_free_0((unsigned __int64)v2[0]);
}
