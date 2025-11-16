// Function: sub_AD6220
// Address: 0xad6220
//
__int64 __fastcall sub_AD6220(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  unsigned __int64 v3; // rax
  unsigned __int8 *v4; // r8
  int v5; // edx
  __int64 v7; // [rsp+8h] [rbp-18h]

  v2 = a1;
  if ( (unsigned int)*(unsigned __int8 *)(a1 + 8) - 17 <= 1 )
    v2 = **(_QWORD **)(a1 + 16);
  v3 = sub_ACCFD0(*(__int64 **)a1, a2);
  v4 = (unsigned __int8 *)v3;
  if ( *(_BYTE *)(v2 + 8) == 14 )
    v4 = (unsigned __int8 *)sub_AD4C70(v3, (__int64 **)v2, 0);
  v5 = *(unsigned __int8 *)(a1 + 8);
  if ( (unsigned int)(v5 - 17) > 1 )
    return (__int64)v4;
  BYTE4(v7) = (_BYTE)v5 == 18;
  LODWORD(v7) = *(_DWORD *)(a1 + 32);
  return sub_AD5E10(v7, v4);
}
