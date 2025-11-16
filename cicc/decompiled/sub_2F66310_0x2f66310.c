// Function: sub_2F66310
// Address: 0x2f66310
//
__int64 __fastcall sub_2F66310(__int64 a1)
{
  int v1; // edx
  __int64 result; // rax
  int v3; // eax
  int v4; // edx
  int v5; // eax

  v1 = *(_DWORD *)(a1 + 8);
  result = 0;
  if ( (unsigned int)(v1 - 1) > 0x3FFFFFFE )
  {
    v3 = *(_DWORD *)(a1 + 12);
    *(_DWORD *)(a1 + 12) = v1;
    v4 = *(_DWORD *)(a1 + 16);
    *(_BYTE *)(a1 + 26) ^= 1u;
    *(_DWORD *)(a1 + 8) = v3;
    v5 = *(_DWORD *)(a1 + 20);
    *(_DWORD *)(a1 + 20) = v4;
    *(_DWORD *)(a1 + 16) = v5;
    return 1;
  }
  return result;
}
