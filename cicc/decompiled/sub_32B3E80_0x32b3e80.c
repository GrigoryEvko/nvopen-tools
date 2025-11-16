// Function: sub_32B3E80
// Address: 0x32b3e80
//
void __fastcall sub_32B3E80(__int64 a1, __int64 a2, char a3, char a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rax
  __int64 v7; // [rsp-20h] [rbp-20h] BYREF

  if ( *(_DWORD *)(a2 + 24) != 328 && (!a4 || *(_DWORD *)(a2 + 88) != -2) )
  {
    if ( a3 )
    {
      v7 = a2;
      sub_32B3B20(a1 + 568, &v7);
      if ( *(int *)(a2 + 88) >= 0 )
        return;
    }
    else if ( *(int *)(a2 + 88) >= 0 )
    {
      return;
    }
    *(_DWORD *)(a2 + 88) = *(_DWORD *)(a1 + 48);
    v6 = *(unsigned int *)(a1 + 48);
    if ( v6 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 52) )
    {
      sub_C8D5F0(a1 + 40, (const void *)(a1 + 56), v6 + 1, 8u, a5, a6);
      v6 = *(unsigned int *)(a1 + 48);
    }
    *(_QWORD *)(*(_QWORD *)(a1 + 40) + 8 * v6) = a2;
    ++*(_DWORD *)(a1 + 48);
  }
}
