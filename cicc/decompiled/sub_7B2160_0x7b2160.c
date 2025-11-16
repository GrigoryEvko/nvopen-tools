// Function: sub_7B2160
// Address: 0x7b2160
//
_DWORD *__fastcall sub_7B2160(
        char *a1,
        int a2,
        unsigned int a3,
        char a4,
        char a5,
        char a6,
        int a7,
        int a8,
        int a9,
        _DWORD *a10)
{
  _DWORD *result; // rax
  int v15; // r12d
  const char *v16; // rax
  int v17; // [rsp+0h] [rbp-60h] BYREF
  int v18; // [rsp+4h] [rbp-5Ch] BYREF
  char *v19; // [rsp+8h] [rbp-58h] BYREF
  char *v20; // [rsp+10h] [rbp-50h] BYREF
  FILE *stream; // [rsp+18h] [rbp-48h] BYREF
  __int64 v22; // [rsp+20h] [rbp-40h] BYREF
  __int64 v23[7]; // [rsp+28h] [rbp-38h] BYREF

  v22 = 0;
  v17 = 0;
  if ( a10 )
    *a10 = 0;
  result = (_DWORD *)sub_7B09E0(a1, a2, a3, a4, a8, 0, a5, a9, &v19, &v20, (__int64 *)&stream, &v17, &v18, v23);
  if ( (_DWORD)result )
  {
    if ( !v17 )
    {
      if ( !a3 || !sub_7AFEF0(v19, &v22, 1, 1) )
        return sub_7B1C00((__int64)stream, (__int64)a1, (__int64)v20, v19, a3, a4, a5, a6, a7, v18, v23[0], v22);
      if ( !v17 )
        fclose(stream);
    }
    if ( a10 )
      *a10 = 1;
    if ( HIDWORD(qword_4D04914) )
    {
      v15 = dword_4F17FD8;
      v16 = (const char *)sub_723260(v20);
      fprintf(qword_4F07510, "%*s%s\n", v15, byte_3F871B3, v16);
    }
    dword_4D03CC0[0] = 1;
    return dword_4D03CC0;
  }
  return result;
}
