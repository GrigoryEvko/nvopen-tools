// Function: sub_131C8A0
// Address: 0x131c8a0
//
void *__fastcall sub_131C8A0(void *s, __int64 a2, char a3)
{
  size_t v3; // rax
  void *result; // rax
  int v5; // ecx
  size_t v6; // rax

  if ( a3 )
  {
    v6 = sub_131C890(a2);
    return memset(s, 0, v6);
  }
  else
  {
    v3 = sub_131C890(a2);
    result = memset(s, 255, v3);
    v5 = -*(_DWORD *)a2 & 0x3F;
    if ( v5 )
    {
      result = *(void **)(a2 + 8);
      *((_QWORD *)s + (_QWORD)result - 1) >>= v5;
    }
  }
  return result;
}
