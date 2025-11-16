// Function: sub_16E3D60
// Address: 0x16e3d60
//
__int64 __fastcall sub_16E3D60(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, int a6)
{
  __int64 v6; // rcx
  __int64 result; // rax
  int v8; // edx

  v6 = *(_QWORD *)(a1 + 32);
  result = *(unsigned int *)(a1 + 40);
  v8 = *(_DWORD *)(v6 + 4 * result - 4);
  if ( v8 == 2 )
  {
    result = (unsigned int)(result - 1);
    *(_DWORD *)(a1 + 40) = result;
    if ( (unsigned int)result >= *(_DWORD *)(a1 + 44) )
    {
      sub_16CD150(a1 + 32, (const void *)(a1 + 48), 0, 4, a5, a6);
      v6 = *(_QWORD *)(a1 + 32);
      result = *(unsigned int *)(a1 + 40);
    }
    *(_DWORD *)(v6 + 4 * result) = 3;
    ++*(_DWORD *)(a1 + 40);
  }
  else if ( v8 == 4 )
  {
    result = (unsigned int)(result - 1);
    *(_DWORD *)(a1 + 40) = result;
    if ( (unsigned int)result >= *(_DWORD *)(a1 + 44) )
    {
      sub_16CD150(a1 + 32, (const void *)(a1 + 48), 0, 4, a5, a6);
      v6 = *(_QWORD *)(a1 + 32);
      result = *(unsigned int *)(a1 + 40);
    }
    *(_DWORD *)(v6 + 4 * result) = 5;
    ++*(_DWORD *)(a1 + 40);
  }
  return result;
}
