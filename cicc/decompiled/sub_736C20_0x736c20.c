// Function: sub_736C20
// Address: 0x736c20
//
__int64 __fastcall sub_736C20(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 result; // rax
  __int64 v4; // rdi

  result = *(unsigned int *)(a1 + 152);
  if ( !(_DWORD)result )
  {
    v4 = *(_QWORD *)(a1 + 120);
    if ( *(char *)(v4 + 142) >= 0 && *(_BYTE *)(v4 + 140) == 12 )
      return sub_8D4AB0(v4, a2, a3);
    else
      return *(unsigned int *)(v4 + 136);
  }
  return result;
}
