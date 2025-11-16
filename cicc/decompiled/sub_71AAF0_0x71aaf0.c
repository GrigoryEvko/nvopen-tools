// Function: sub_71AAF0
// Address: 0x71aaf0
//
__int64 __fastcall sub_71AAF0(__int64 a1, int a2, int a3, unsigned int a4, _DWORD *a5, __int64 a6)
{
  __int64 v8; // r9
  unsigned int v9; // r12d
  __int64 v11; // r13
  __m128i v13[4]; // [rsp+10h] [rbp-40h] BYREF

  v13[0] = 0u;
  v9 = sub_7A2650(a1, a4, a5, a6, v13, 1);
  if ( v9 )
  {
    if ( a2 )
      sub_71AAB0(a6, a1);
  }
  else
  {
    v11 = *(_QWORD *)(a1 + 56);
    if ( v11 )
    {
      if ( (*(_BYTE *)(v11 + 193) & 4) != 0 && (unsigned int)sub_6E5270(v11, a6, a5, v13) )
      {
        v9 = 1;
      }
      else if ( a3 && (unsigned int)sub_6F50A0(v11, 0, 0, v13, (__int64)a5, v8) )
      {
        v9 = 1;
        sub_72C970(a6);
      }
    }
  }
  sub_67E3D0(v13);
  return v9;
}
