// Function: sub_2EC9080
// Address: 0x2ec9080
//
__int64 __fastcall sub_2EC9080(__int64 a1, __int64 a2, __int64 a3, char a4, int *a5)
{
  unsigned int v5; // r9d
  unsigned int v7; // eax
  unsigned int v8; // edx
  int v9; // ecx

  v5 = 1;
  v7 = *(_DWORD *)(a3 + 164);
  v8 = *(_DWORD *)(a1 + 40);
  if ( v8 >= v7 )
  {
    v5 = 0;
    if ( v7 )
    {
      if ( a4 )
      {
        v9 = sub_2EC8CA0(a3);
        *a5 = v9;
        v7 = *(_DWORD *)(a3 + 164);
        v8 = *(_DWORD *)(a1 + 40);
      }
      else
      {
        v9 = *a5;
      }
      LOBYTE(v5) = v9 + v7 > v8;
    }
  }
  return v5;
}
