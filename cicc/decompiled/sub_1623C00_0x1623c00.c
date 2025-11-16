// Function: sub_1623C00
// Address: 0x1623c00
//
unsigned __int64 __fastcall sub_1623C00(unsigned int *a1, unsigned int a2, __int64 a3)
{
  unsigned int v3; // eax
  __int64 v4; // rdx
  unsigned __int64 result; // rax
  unsigned __int8 *v6; // rsi
  __int64 v7; // rsi
  _QWORD v8[3]; // [rsp+8h] [rbp-18h] BYREF

  v8[0] = a3;
  sub_1623A60((__int64)v8, a3, 2);
  v3 = a1[2];
  if ( v3 >= a1[3] )
  {
    sub_1623260(a1, 0);
    v3 = a1[2];
  }
  v4 = *(_QWORD *)a1 + 16LL * v3;
  if ( v4 )
  {
    result = a2;
    *(_DWORD *)v4 = a2;
    v6 = (unsigned __int8 *)v8[0];
    *(_QWORD *)(v4 + 8) = v8[0];
    if ( v6 )
      result = sub_1623210((__int64)v8, v6, v4 + 8);
    ++a1[2];
  }
  else
  {
    v7 = v8[0];
    result = v3 + 1;
    a1[2] = result;
    if ( v7 )
      return sub_161E7C0((__int64)v8, v7);
  }
  return result;
}
