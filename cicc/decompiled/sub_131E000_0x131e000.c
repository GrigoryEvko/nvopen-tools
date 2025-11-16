// Function: sub_131E000
// Address: 0x131e000
//
__int64 __fastcall sub_131E000(char *a1, __int64 a2, __int64 a3, _BYTE *a4, _BOOL8 *a5, char *a6, __int64 a7)
{
  char v9; // r13
  __int64 result; // rax
  char v11; // r15
  _BOOL8 v12; // rax
  _BOOL8 v13; // r8
  _BOOL4 v14; // edi
  unsigned int v15; // eax
  __int64 v16; // rdx
  _BYTE v17[33]; // [rsp+1Fh] [rbp-21h]

  v9 = *a1;
  v17[0] = *a1;
  if ( a6 )
  {
    result = 22;
    if ( a7 != 1 )
      return result;
    v11 = *a6;
    if ( v9 == 1 || !v11 )
    {
      if ( v11 != 1 && v9 )
        sub_1311F30((__int64)a1);
    }
    else
    {
      sub_13124F0((__int64)a1);
    }
    *a1 = v11;
    sub_1313A40(a1);
  }
  if ( !a4 || !a5 )
    return 0;
  v12 = *a5;
  if ( *a5 )
  {
    *a4 = v9;
    return 0;
  }
  else
  {
    v13 = v12;
    v14 = v12;
    if ( v12 )
    {
      v15 = 0;
      do
      {
        v16 = v15++;
        a4[v16] = v17[v16];
      }
      while ( v15 < v14 );
    }
    *a5 = v13;
    return 22;
  }
}
