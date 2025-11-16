// Function: sub_1691F20
// Address: 0x1691f20
//
__int64 __fastcall sub_1691F20(
        __int64 a1,
        __int64 a2,
        _BYTE *a3,
        __int64 a4,
        unsigned int *a5,
        unsigned int *a6,
        int a7,
        int a8)
{
  unsigned int v8; // r15d
  unsigned int v13; // r14d
  __int64 *v14; // rsi
  const char *v15; // rdi
  unsigned int v17; // r9d
  unsigned int v20[13]; // [rsp+1Ch] [rbp-34h] BYREF

  v8 = a4;
  sub_168FBD0(a1, a3, &a3[8 * a4]);
  *a6 = 0;
  v20[0] = 0;
  *a5 = 0;
  if ( v8 )
  {
    v13 = 0;
    while ( 1 )
    {
      while ( 1 )
      {
        v15 = *(const char **)(*(_QWORD *)(a1 + 184) + 8LL * v13);
        if ( v15 )
        {
          if ( strlen(v15) )
            break;
        }
        v20[0] = ++v13;
        if ( v8 <= v13 )
          return a1;
      }
      v14 = (__int64 *)sub_1691970(a2, (const char **)a1, v20, a7, a8);
      if ( !v14 )
        break;
      sub_1690030(a1, v14);
      v13 = v20[0];
      if ( v8 <= v20[0] )
        return a1;
    }
    v17 = v20[0] + ~v13;
    *a5 = v13;
    *a6 = v17;
  }
  return a1;
}
