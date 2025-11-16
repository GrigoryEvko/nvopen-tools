// Function: sub_15160C0
// Address: 0x15160c0
//
__int64 __fastcall sub_15160C0(__int64 *a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rcx
  int v4; // edi
  unsigned int v5; // edx
  __int64 *v6; // rax
  __int64 v7; // r8
  __int64 result; // rax
  int v9; // edx
  int v10; // eax
  int v11; // r9d

  v2 = *a1;
  if ( (*(_BYTE *)(*a1 + 712) & 1) != 0 )
  {
    v3 = v2 + 720;
    v4 = 15;
  }
  else
  {
    v9 = *(_DWORD *)(v2 + 728);
    v3 = *(_QWORD *)(v2 + 720);
    result = 0;
    if ( !v9 )
      return result;
    v4 = v9 - 1;
  }
  v5 = v4 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v6 = (__int64 *)(v3 + 16LL * v5);
  v7 = *v6;
  if ( a2 == *v6 )
    return v6[1];
  v10 = 1;
  while ( v7 != -8 )
  {
    v11 = v10 + 1;
    v5 = v4 & (v10 + v5);
    v6 = (__int64 *)(v3 + 16LL * v5);
    v7 = *v6;
    if ( a2 == *v6 )
      return v6[1];
    v10 = v11;
  }
  return 0;
}
