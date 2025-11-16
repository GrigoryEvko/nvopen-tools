// Function: sub_16A8E60
// Address: 0x16a8e60
//
bool __fastcall sub_16A8E60(__int64 a1, unsigned int a2)
{
  bool result; // al
  bool v3; // [rsp+Fh] [rbp-21h]
  const void *v4; // [rsp+10h] [rbp-20h] BYREF
  unsigned int v5; // [rsp+18h] [rbp-18h]

  sub_16A8CB0((__int64)&v4, a1, a2);
  if ( *(_DWORD *)(a1 + 8) <= 0x40u )
    result = *(_QWORD *)a1 == (_QWORD)v4;
  else
    result = sub_16A5220(a1, &v4);
  if ( v5 > 0x40 )
  {
    if ( v4 )
    {
      v3 = result;
      j_j___libc_free_0_0(v4);
      return v3;
    }
  }
  return result;
}
