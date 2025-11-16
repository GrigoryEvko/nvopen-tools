// Function: sub_F70890
// Address: 0xf70890
//
__int64 __fastcall sub_F70890(__int64 a1, __int64 a2, __int64 a3, unsigned __int64 a4)
{
  __int64 result; // rax
  unsigned __int64 v7; // r14
  unsigned int v8; // [rsp+4h] [rbp-2Ch] BYREF
  __int64 v9; // [rsp+8h] [rbp-28h]

  v8 = 0;
  result = sub_F6EC60(a1, &v8);
  v9 = result;
  if ( BYTE4(result) )
  {
    v7 = (unsigned int)v9 % a4;
    sub_F6ED70(a2, (unsigned int)v9 / a4, v8);
    return sub_F6ED70(a3, v7, v8);
  }
  return result;
}
