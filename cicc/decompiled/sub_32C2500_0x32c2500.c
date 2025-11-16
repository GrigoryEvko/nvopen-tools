// Function: sub_32C2500
// Address: 0x32c2500
//
void __fastcall sub_32C2500(__int64 *a1, __int64 a2)
{
  __int64 v2; // r12
  __int64 v3; // r8
  __int64 v4; // r9
  __int64 v5; // rax
  __int64 v6; // [rsp-20h] [rbp-20h] BYREF

  if ( *(_DWORD *)(a2 + 24) != 328 )
  {
    v2 = *a1;
    v6 = a2;
    sub_32B3B20(v2 + 568, &v6);
    if ( *(int *)(a2 + 88) < 0 )
    {
      *(_DWORD *)(a2 + 88) = *(_DWORD *)(v2 + 48);
      v5 = *(unsigned int *)(v2 + 48);
      if ( v5 + 1 > (unsigned __int64)*(unsigned int *)(v2 + 52) )
      {
        sub_C8D5F0(v2 + 40, (const void *)(v2 + 56), v5 + 1, 8u, v3, v4);
        v5 = *(unsigned int *)(v2 + 48);
      }
      *(_QWORD *)(*(_QWORD *)(v2 + 40) + 8 * v5) = a2;
      ++*(_DWORD *)(v2 + 48);
    }
  }
}
