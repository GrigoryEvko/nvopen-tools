// Function: sub_1F0FD40
// Address: 0x1f0fd40
//
__int64 __fastcall sub_1F0FD40(__int64 *a1)
{
  __int64 v1; // r10
  __int64 v2; // r11
  __int64 *v3; // rdx
  __int64 v4; // rax
  unsigned __int64 v5; // r8
  __int64 v6; // rcx
  int v7; // r9d
  __int64 v8; // rax
  __int64 result; // rax

  v1 = *a1;
  v2 = a1[1];
  v3 = a1 - 2;
  v4 = *(a1 - 2);
  v5 = *a1 & 0xFFFFFFFFFFFFFFF8LL;
  v6 = (*a1 >> 1) & 3;
  v7 = v6;
  while ( 1 )
  {
    result = *(_DWORD *)((v4 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v4 >> 1) & 3;
    if ( (*(_DWORD *)(v5 + 24) | (unsigned int)v6) >= (unsigned int)result )
      break;
    v8 = *v3;
    a1 = v3;
    v3 -= 2;
    LODWORD(v6) = v7;
    v3[4] = v8;
    v3[5] = v3[3];
    v4 = *v3;
  }
  *a1 = v1;
  a1[1] = v2;
  return result;
}
