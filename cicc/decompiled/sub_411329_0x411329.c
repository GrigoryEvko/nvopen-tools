// Function: sub_411329
// Address: 0x411329
//
__int64 __fastcall sub_411329(
        __int64 a1,
        __int64 a2,
        int a3,
        int a4,
        const char **a5,
        __int64 a6,
        int a7,
        const char **a8)
{
  __int64 v8; // r10
  int v13; // ecx
  int v14; // r8d
  int v15; // r9d
  int v16; // edx
  int v17; // ecx
  int v18; // r8d
  int v19; // r9d
  int v20; // edx
  int v21; // ecx
  int v22; // r8d
  int v23; // r9d
  __int64 v25; // [rsp-8h] [rbp-30h]

  v25 = v8;
  if ( *(_DWORD *)a1 > 1u )
  {
    if ( *(_DWORD *)a1 == 2 )
    {
      sub_130F150(a1);
      sub_130F0B0(a1, (unsigned int)"%s: ", a3, v13, v14, v15);
      sub_40EBBB(a1, 2, -1, a4, a5);
      if ( a6 )
      {
        sub_130F0B0(a1, (unsigned int)" (%s: ", a6, v17, v18, v19);
        sub_40EBBB(a1, 2, -1, a7, a8);
        sub_130F0B0(a1, (unsigned int)")", v20, v21, v22, v23);
      }
      sub_130F0B0(a1, (unsigned int)"\n", v16, v17, v18, v19);
    }
  }
  else
  {
    sub_130F450(a1, a2);
    if ( *(_DWORD *)a1 <= 1u )
    {
      sub_130F270(a1);
      sub_40EBBB(a1, 2, -1, a4, a5);
    }
  }
  *(_BYTE *)(a1 + 28) = 1;
  return v25;
}
