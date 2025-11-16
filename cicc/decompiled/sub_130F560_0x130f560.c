// Function: sub_130F560
// Address: 0x130f560
//
__int64 __fastcall sub_130F560(unsigned int *a1, const char *a2)
{
  __int64 result; // rax
  int v3; // ecx
  const char *v4; // r14
  int v5; // r13d
  const char *v6; // rcx
  bool v7; // cc
  int v8; // [rsp-3Ch] [rbp-3Ch]

  result = *a1;
  if ( (unsigned int)result <= 1 )
  {
    if ( *((_BYTE *)a1 + 29) )
    {
      *((_BYTE *)a1 + 29) = 0;
    }
    else
    {
      if ( *((_BYTE *)a1 + 28) )
      {
        sub_130F0B0((__int64)a1, ",");
        LODWORD(result) = *a1;
      }
      if ( (_DWORD)result == 1 )
        goto LABEL_18;
      sub_130F0B0((__int64)a1, "\n");
      v3 = a1[6];
      LODWORD(result) = *a1;
      v8 = v3;
      if ( *a1 )
      {
        v8 = 2 * v3;
        if ( 2 * v3 <= 0 )
          goto LABEL_13;
        v4 = " ";
      }
      else
      {
        v4 = "\t";
        if ( v3 <= 0 )
          goto LABEL_14;
      }
      v5 = 0;
      do
      {
        sub_130F0B0((__int64)a1, "%s", v4);
        ++v5;
      }
      while ( v5 < v8 );
      LODWORD(result) = *a1;
    }
LABEL_13:
    if ( (_DWORD)result != 1 )
    {
LABEL_14:
      v6 = " ";
      goto LABEL_15;
    }
LABEL_18:
    v6 = byte_3F871B3;
LABEL_15:
    result = sub_130F0B0((__int64)a1, "\"%s\":%s", a2, v6);
    v7 = *a1 <= 1;
    *((_BYTE *)a1 + 29) = 1;
    if ( v7 )
    {
      *((_BYTE *)a1 + 29) = 0;
      result = sub_130F0B0((__int64)a1, "{");
      ++a1[6];
      *((_BYTE *)a1 + 28) = 0;
    }
  }
  return result;
}
