// Function: sub_703C10
// Address: 0x703c10
//
__int64 __fastcall sub_703C10(_BYTE *a1)
{
  const char *v1; // r13
  unsigned int v2; // r14d
  unsigned int v3; // r15d
  unsigned int v4; // r12d
  __int64 v5; // rbx
  int v6; // eax
  __int64 result; // rax
  __int64 v8; // [rsp+8h] [rbp-38h]

  v1 = &a1[*a1 == 37];
  if ( !(_DWORD)nmemb )
    return 58;
  v2 = nmemb;
  v3 = 0;
  v8 = qword_4D03AC0;
  while ( 1 )
  {
    while ( 1 )
    {
      v4 = (v3 + v2) >> 1;
      v5 = v8 + 16LL * v4;
      v6 = strcmp(v1, *(const char **)v5);
      if ( v6 <= 0 )
        break;
      v3 = v4 + 1;
      if ( v2 <= v4 + 1 )
        return 58;
    }
    if ( !v6 )
      break;
    v2 = (v3 + v2) >> 1;
    if ( v4 <= v3 )
      return 58;
  }
  result = *(unsigned __int8 *)(v5 + 8);
  if ( !(_BYTE)result )
    return 58;
  return result;
}
