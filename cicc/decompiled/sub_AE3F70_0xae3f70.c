// Function: sub_AE3F70
// Address: 0xae3f70
//
unsigned __int64 __fastcall sub_AE3F70(_QWORD *a1, _BYTE *a2, __int64 a3)
{
  unsigned __int64 result; // rax
  unsigned __int64 v5; // [rsp+0h] [rbp-30h] BYREF
  __int64 v6[5]; // [rsp+8h] [rbp-28h] BYREF

  sub_AE1D50((__int64)a1);
  sub_AE3AA0(&v5, a1, a2, a3);
  result = v5 & 0xFFFFFFFFFFFFFFFELL;
  if ( (v5 & 0xFFFFFFFFFFFFFFFELL) != 0 )
  {
    v5 = 0;
    v6[0] = result | 1;
    sub_C641D0(v6, 1);
  }
  return result;
}
