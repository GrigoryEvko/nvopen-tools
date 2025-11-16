// Function: sub_131E100
// Address: 0x131e100
//
__int64 __fastcall sub_131E100(__int64 a1, _QWORD *a2, const char *a3, _QWORD *a4, __int64 a5, unsigned __int64 *a6)
{
  char *v6; // r15
  size_t v7; // r12
  __int64 v8; // r13
  __int64 v9; // r14
  __int64 v10; // rbx
  const char *v11; // r15
  char v12; // al
  __int64 result; // rax
  unsigned __int64 v14; // rbx
  __int64 v15; // rax
  char *s1; // [rsp+20h] [rbp-50h]
  unsigned __int64 v20; // [rsp+28h] [rbp-48h]
  __int64 v22; // [rsp+38h] [rbp-38h]

  v6 = (char *)a3;
  v22 = (__int64)strchr(a3, 46);
  if ( !v22 )
    v22 = __rawmemchr();
  v7 = v22 - (_QWORD)v6;
  if ( (char *)v22 == v6 )
    return 2;
  if ( *a6 )
  {
    v20 = 0;
    while ( 1 )
    {
      if ( *(_BYTE *)a2[3] )
      {
        v8 = a2[2];
        if ( !v8 )
          return 2;
        s1 = v6;
        v9 = a2[3];
        v10 = 0;
        while ( 1 )
        {
          v11 = *(const char **)(v9 + 8);
          if ( strlen(v11) == v7 && !strncmp(s1, v11, v7) )
            break;
          ++v10;
          v9 += 40;
          if ( v10 == v8 )
            return 2;
        }
        *(_QWORD *)(a5 + 8 * v20) = v10;
        if ( (_QWORD *)v9 == a2 )
          return 2;
        a2 = (_QWORD *)v9;
      }
      else
      {
        v14 = sub_130AAB0(v6, 0, 0xAu);
        if ( v14 == -1 )
          return 2;
        v15 = a2[3];
        if ( *(_BYTE *)v15 )
          BUG();
        a2 = (_QWORD *)(*(__int64 (__fastcall **)(__int64, __int64, unsigned __int64, unsigned __int64))(v15 + 8))(
                         a1,
                         a5,
                         *a6,
                         v14);
        if ( !a2 )
          return 2;
        *(_QWORD *)(a5 + 8 * v20) = v14;
      }
      v12 = *(_BYTE *)v22;
      if ( a2[4] )
        break;
      if ( !v12 )
        goto LABEL_28;
      v6 = (char *)(v22 + 1);
      v22 = (__int64)strchr((const char *)(v22 + 1), 46);
      if ( !v22 )
        v22 = __rawmemchr();
      ++v20;
      v7 = v22 - (_QWORD)v6;
      if ( *a6 <= v20 )
        goto LABEL_19;
    }
    if ( v12 )
      return 2;
LABEL_28:
    *a6 = v20 + 1;
  }
LABEL_19:
  result = 0;
  if ( a4 )
    *a4 = a2;
  return result;
}
