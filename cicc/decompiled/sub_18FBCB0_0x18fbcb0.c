// Function: sub_18FBCB0
// Address: 0x18fbcb0
//
__int64 __fastcall sub_18FBCB0(unsigned __int8 *a1)
{
  __int64 result; // rax
  __int64 v2; // rdi
  char v3; // dl
  unsigned int v4; // ecx
  int v5; // eax

  result = *a1;
  if ( (_BYTE)result )
  {
    result = 0;
    if ( *((_DWORD *)a1 + 4) <= 1u )
      return a1[24] ^ 1u;
  }
  else
  {
    v2 = *((_QWORD *)a1 + 4);
    v3 = *(_BYTE *)(v2 + 16);
    if ( v3 == 54 || v3 == 55 )
    {
      v4 = *(unsigned __int16 *)(v2 + 18);
      if ( ((v4 >> 7) & 6) == 0 )
        return !(v4 & 1);
    }
    else
    {
      LOBYTE(v5) = sub_15F32D0(v2);
      return v5 ^ 1u;
    }
  }
  return result;
}
