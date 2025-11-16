// Function: sub_21E8CD0
// Address: 0x21e8cd0
//
unsigned __int64 __fastcall sub_21E8CD0(__int64 a1, unsigned int a2, __int64 a3, const char *a4)
{
  __int64 v5; // r8
  bool v6; // zf
  char *v7; // rsi
  unsigned __int64 result; // rax
  char *v9; // r13
  size_t v10; // rax
  __int64 v11; // rcx
  size_t v12; // rdx
  size_t v13; // rax
  __int64 v14; // rcx
  __int64 v15; // rsi
  __int64 v16; // rsi

  v5 = *(_QWORD *)(*(_QWORD *)(a1 + 16) + 16LL * a2 + 8);
  if ( strcmp(a4, "scaleD") )
  {
    if ( !strcmp(a4, "negA") )
    {
      v9 = "-1";
      if ( (v5 & 2) == 0 )
        v9 = "1";
      v10 = strlen(v9);
      v11 = *(_QWORD *)(a3 + 24);
      v12 = v10;
      result = *(_QWORD *)(a3 + 16) - v11;
      if ( v12 > result )
        return sub_16E7EE0(a3, v9, v12);
      if ( (_DWORD)v12 )
      {
        LODWORD(result) = 0;
        do
        {
          v16 = (unsigned int)result;
          result = (unsigned int)(result + 1);
          *(_BYTE *)(v11 + v16) = v9[v16];
        }
        while ( (unsigned int)result < (unsigned int)v12 );
      }
    }
    else
    {
      if ( strcmp(a4, "negB") )
      {
        if ( !strcmp(a4, "transA") )
          v6 = (v5 & 8) == 0;
        else
          v6 = (v5 & 0x10) == 0;
        goto LABEL_3;
      }
      v9 = "-1";
      if ( (v5 & 4) == 0 )
        v9 = "1";
      v13 = strlen(v9);
      v14 = *(_QWORD *)(a3 + 24);
      v12 = v13;
      result = *(_QWORD *)(a3 + 16) - v14;
      if ( v12 > result )
        return sub_16E7EE0(a3, v9, v12);
      if ( (_DWORD)v12 )
      {
        LODWORD(result) = 0;
        do
        {
          v15 = (unsigned int)result;
          result = (unsigned int)(result + 1);
          *(_BYTE *)(v14 + v15) = v9[v15];
        }
        while ( (unsigned int)result < (unsigned int)v12 );
      }
    }
    *(_QWORD *)(a3 + 24) += v12;
    return result;
  }
  v6 = (v5 & 1) == 0;
LABEL_3:
  v7 = "1";
  if ( v6 )
    v7 = "0";
  result = *(_QWORD *)(a3 + 24);
  if ( *(_QWORD *)(a3 + 16) == result )
    return sub_16E7EE0(a3, v7, 1u);
  *(_BYTE *)result = *v7;
  ++*(_QWORD *)(a3 + 24);
  return result;
}
