// Function: sub_729F20
// Address: 0x729f20
//
_BOOL8 __fastcall sub_729F20(unsigned int a1)
{
  __int64 v1; // rbx
  _BOOL8 result; // rax
  const char **v3; // rdx
  int v4; // [rsp+8h] [rbp-18h] BYREF
  _DWORD v5[3]; // [rsp+Ch] [rbp-14h] BYREF

  v1 = sub_729B10(a1, &v4, v5, 0);
  result = 0;
  if ( v1 )
  {
    v3 = (const char **)sub_729EA0();
    result = 1;
    if ( (*(_BYTE *)(v1 + 72) & 4) == 0 )
    {
      result = 0;
      if ( !*(_QWORD *)(v1 + 8) )
        return strcmp(*(const char **)v1, *v3) != 0;
    }
  }
  return result;
}
