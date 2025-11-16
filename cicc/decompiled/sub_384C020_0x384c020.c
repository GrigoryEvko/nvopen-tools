// Function: sub_384C020
// Address: 0x384c020
//
__int64 __fastcall sub_384C020(_QWORD *a1, __int64 a2)
{
  _DWORD *v2; // rax
  unsigned int v3; // r8d

  v2 = (_DWORD *)(*a1 + 16LL * *(unsigned int *)(a2 + 24));
  v3 = *(unsigned __int8 *)v2;
  if ( !(_BYTE)v3 )
    return v3;
  v3 = 0;
  if ( v2[3] != v2[2] )
    return v3;
  LOBYTE(v3) = v2[1] == *(_DWORD *)(a2 + 72);
  return v3;
}
