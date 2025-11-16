// Function: sub_B96FD0
// Address: 0xb96fd0
//
__int64 __fastcall sub_B96FD0(__int64 a1, __int64 a2)
{
  unsigned __int8 v2; // al
  __int64 *v3; // rbx
  __int64 *v4; // r14
  __int64 v5; // r12
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 result; // rax

  v2 = *(_BYTE *)(a1 - 16);
  if ( (v2 & 2) != 0 )
  {
    v3 = *(__int64 **)(a1 - 32);
    v4 = &v3[*(unsigned int *)(a1 - 24)];
  }
  else
  {
    v3 = (__int64 *)(a1 - 8LL * ((v2 >> 2) & 0xF) - 16);
    v4 = &v3[(*(_WORD *)(a1 - 16) >> 6) & 0xF];
  }
  for ( ; v4 != v3; ++v3 )
  {
    v5 = *v3;
    if ( *v3 )
    {
      sub_B91220((__int64)v3, *v3);
      *v3 = v5;
      a2 = v5;
      sub_B96E90((__int64)v3, v5, a1 & 0xFFFFFFFFFFFFFFFCLL | 1);
    }
  }
  *(_BYTE *)(a1 + 1) &= 0x80u;
  sub_B91600(a1);
  result = *(unsigned int *)(a1 - 8);
  if ( !(_DWORD)result )
    return sub_B93110(a1, a2, v6, v7, v8);
  return result;
}
