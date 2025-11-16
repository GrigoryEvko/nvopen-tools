// Function: sub_2F57F90
// Address: 0x2f57f90
//
__int64 __fastcall sub_2F57F90(_QWORD *a1, __int64 a2, __int64 a3, unsigned int a4, _BYTE *a5, __int64 a6)
{
  int v8; // eax
  __int64 v11; // rax
  unsigned int v12; // edx
  __int64 v13; // r9
  unsigned int v15; // [rsp+14h] [rbp-3Ch] BYREF
  _QWORD v16[7]; // [rsp+18h] [rbp-38h] BYREF

  v8 = *(_DWORD *)(a1[115] + 8LL * (*(_DWORD *)(a2 + 112) & 0x7FFFFFFF));
  if ( v8 != 4 )
  {
    if ( v8 <= 1 )
    {
      sub_2FB1E90(a1[124], a2);
      v11 = a1[3618];
      v15 = 0;
      v16[0] = v11;
      v12 = sub_2F579F0((__int64)a1, a2, a3, (__int64)v16, &v15, 1);
      if ( v12 != -1 )
      {
        sub_2F53540((__int64)a1, a2, v12, 0, a6, v13);
        return 0;
      }
    }
    return a4;
  }
  if ( *(float *)(a2 + 116) == INFINITY )
    return a4;
  sub_2FB1E90(a1[124], a2);
  if ( a1[3618] <= (unsigned __int64)sub_2F525E0((__int64)a1) )
    return a4;
  *a5 = 1;
  return 0;
}
