// Function: sub_CA89D0
// Address: 0xca89d0
//
_QWORD *__fastcall sub_CA89D0(__int64 ***a1, __int64 a2, __int64 a3, int a4)
{
  __int64 v4; // rax
  unsigned __int64 v6; // [rsp+0h] [rbp-10h] BYREF
  __int64 v7; // [rsp+8h] [rbp-8h]

  if ( a2 )
  {
    v4 = *(_QWORD *)(a2 + 24);
    v6 = *(_QWORD *)(a2 + 16);
    v7 = v4;
  }
  else
  {
    v6 = 0;
    v7 = 0;
  }
  return sub_CA8990(a1, &v6, a3, a4);
}
