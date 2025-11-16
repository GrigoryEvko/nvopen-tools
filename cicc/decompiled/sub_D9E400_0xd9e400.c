// Function: sub_D9E400
// Address: 0xd9e400
//
__int64 __fastcall sub_D9E400(__int64 a1)
{
  __int64 result; // rax
  __int64 v2; // rbx
  __int64 v3; // r12
  __int64 v4; // r12
  __int64 v5; // rdi

  if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
  {
    v2 = a1 + 16;
    v3 = 384;
  }
  else
  {
    result = *(unsigned int *)(a1 + 24);
    if ( !(_DWORD)result )
      return result;
    v2 = *(_QWORD *)(a1 + 16);
    v3 = 24 * result;
  }
  v4 = v2 + v3;
  do
  {
    result = *(_QWORD *)v2;
    if ( *(_QWORD *)v2 != -8192 && result != -4096 && *(_DWORD *)(v2 + 16) > 0x40u )
    {
      v5 = *(_QWORD *)(v2 + 8);
      if ( v5 )
        result = j_j___libc_free_0_0(v5);
    }
    v2 += 24;
  }
  while ( v2 != v4 );
  return result;
}
