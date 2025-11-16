// Function: sub_2241CB0
// Address: 0x2241cb0
//
__int64 __fastcall sub_2241CB0(__int64 a1, __int64 a2)
{
  unsigned __int64 v3[2]; // [rsp+0h] [rbp-28h] BYREF
  char v4; // [rsp+10h] [rbp-18h] BYREF

  (*(void (__fastcall **)(unsigned __int64 *))(*(_QWORD *)a2 + 32LL))(v3);
  sub_2257330(a1, v3[0], v3[1]);
  if ( (char *)v3[0] != &v4 )
    j___libc_free_0(v3[0]);
  return a1;
}
