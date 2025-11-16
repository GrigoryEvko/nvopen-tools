// Function: sub_2B5C2D0
// Address: 0x2b5c2d0
//
__int64 __fastcall sub_2B5C2D0(__int64 a1, __int64 *a2)
{
  __int64 result; // rax
  __int64 v3; // r8
  __int64 v4; // r9
  __int64 v5; // r12
  _BYTE v6[64]; // [rsp+0h] [rbp-40h] BYREF

  result = sub_2400480((__int64)v6, a1, a2);
  if ( v6[32] )
  {
    result = *(unsigned int *)(a1 + 40);
    v5 = *a2;
    if ( result + 1 > (unsigned __int64)*(unsigned int *)(a1 + 44) )
    {
      sub_C8D5F0(a1 + 32, (const void *)(a1 + 48), result + 1, 8u, v3, v4);
      result = *(unsigned int *)(a1 + 40);
    }
    *(_QWORD *)(*(_QWORD *)(a1 + 32) + 8 * result) = v5;
    ++*(_DWORD *)(a1 + 40);
  }
  return result;
}
