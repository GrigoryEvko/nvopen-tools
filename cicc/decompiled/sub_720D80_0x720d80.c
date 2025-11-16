// Function: sub_720D80
// Address: 0x720d80
//
int __fastcall sub_720D80(FILE *stream, _DWORD *a2, __int64 a3)
{
  int result; // eax
  int v5; // r12d
  int v6[10]; // [rsp+8h] [rbp-28h] BYREF

  *a2 = unk_4F076FC;
  result = getc(stream);
  if ( result != -1 )
  {
    v5 = result;
    if ( (unsigned int)(result - 254) > 1 && result != 239 )
      return ungetc(result, stream);
    result = getc(stream);
    if ( v5 == 239 && result == 187 )
    {
      result = getc(stream);
      if ( result == 191 )
      {
        *a2 = 1;
        return result;
      }
    }
    else
    {
      if ( v5 == 255 && result == 254 )
      {
        *a2 = 2;
        return result;
      }
      if ( v5 == 254 && result == 255 )
      {
        *a2 = 3;
        return result;
      }
    }
    result = fseek(stream, 0, 0);
    if ( result )
    {
      sub_720D70(v6);
      return sub_685AD0(9u, 1702, a3, v6);
    }
  }
  return result;
}
