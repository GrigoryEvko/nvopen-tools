// Function: sub_1BF18B0
// Address: 0x1bf18b0
//
char *__fastcall sub_1BF18B0(__int64 a1)
{
  int v1; // eax
  const char *v2; // r8
  int v3; // edx

  v1 = *(_DWORD *)(a1 + 8);
  v2 = "loop-vectorize";
  if ( v1 != 1 )
  {
    v3 = *(_DWORD *)(a1 + 40);
    if ( v3 )
    {
      if ( v1 || v3 != -1 )
        return (char *)off_4C6F360;
    }
  }
  return (char *)v2;
}
