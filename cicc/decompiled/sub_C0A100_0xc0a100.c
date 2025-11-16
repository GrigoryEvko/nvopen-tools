// Function: sub_C0A100
// Address: 0xc0a100
//
__int64 __fastcall sub_C0A100(char **a1, _DWORD *a2, _DWORD *a3, _BYTE *a4, size_t a5)
{
  unsigned int v7; // r8d
  char *v8; // rcx
  char *v10; // rdi
  int v12; // eax
  char *v14; // [rsp+8h] [rbp-58h]
  __int64 v15[7]; // [rsp+28h] [rbp-38h] BYREF

  v7 = 1;
  v8 = a1[1];
  if ( (unsigned __int64)v8 >= a5 )
  {
    v10 = *a1;
    if ( !a5 || (v14 = v8, v12 = memcmp(v10, a4, a5), v8 = v14, v7 = 1, !v12) )
    {
      *a1 = &v10[a5];
      a1[1] = &v8[-a5];
      *a2 = sub_C09F10(a4, a5);
      if ( !(unsigned __int8)sub_C93C00(a1, 10, v15) && v15[0] == SLODWORD(v15[0]) )
      {
        *a3 = v15[0];
        return 0;
      }
      else
      {
        return 2;
      }
    }
  }
  return v7;
}
