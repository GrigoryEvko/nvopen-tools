// Function: sub_35F3E90
// Address: 0x35f3e90
//
unsigned __int64 __fastcall sub_35F3E90(__int64 a1, __int64 a2, unsigned int a3, __int64 a4, const char *a5)
{
  __int64 v5; // rdx
  bool v7; // zf
  char *v8; // rsi
  unsigned __int64 result; // rax
  char *v10; // r13
  size_t v11; // rax
  __int64 v12; // rcx
  size_t v13; // rdx
  size_t v14; // rax
  __int64 v15; // rcx
  __int64 v16; // rsi
  __int64 v17; // rsi

  v5 = *(_QWORD *)(*(_QWORD *)(a2 + 16) + 16LL * a3 + 8);
  if ( !a5 )
    goto LABEL_30;
  if ( strcmp(a5, "scaleD") )
  {
    if ( !strcmp(a5, "negA") )
    {
      v10 = "-1";
      if ( (v5 & 2) == 0 )
        v10 = "1";
      v11 = strlen(v10);
      v12 = *(_QWORD *)(a4 + 32);
      v13 = v11;
      result = *(_QWORD *)(a4 + 24) - v12;
      if ( v13 > result )
        return sub_CB6200(a4, (unsigned __int8 *)v10, v13);
      if ( (_DWORD)v13 )
      {
        LODWORD(result) = 0;
        do
        {
          v17 = (unsigned int)result;
          result = (unsigned int)(result + 1);
          *(_BYTE *)(v12 + v17) = v10[v17];
        }
        while ( (unsigned int)result < (unsigned int)v13 );
      }
LABEL_23:
      *(_QWORD *)(a4 + 32) += v13;
      return result;
    }
    if ( !strcmp(a5, "negB") )
    {
      v10 = "-1";
      if ( (v5 & 4) == 0 )
        v10 = "1";
      v14 = strlen(v10);
      v15 = *(_QWORD *)(a4 + 32);
      v13 = v14;
      result = *(_QWORD *)(a4 + 24) - v15;
      if ( v13 > result )
        return sub_CB6200(a4, (unsigned __int8 *)v10, v13);
      if ( (_DWORD)v13 )
      {
        LODWORD(result) = 0;
        do
        {
          v16 = (unsigned int)result;
          result = (unsigned int)(result + 1);
          *(_BYTE *)(v15 + v16) = v10[v16];
        }
        while ( (unsigned int)result < (unsigned int)v13 );
      }
      goto LABEL_23;
    }
    if ( !strcmp(a5, "transA") )
    {
      v7 = (v5 & 8) == 0;
      goto LABEL_4;
    }
    if ( !strcmp(a5, "transB") )
    {
      v7 = (v5 & 0x10) == 0;
      goto LABEL_4;
    }
LABEL_30:
    BUG();
  }
  v7 = (v5 & 1) == 0;
LABEL_4:
  v8 = "1";
  if ( v7 )
    v8 = "0";
  result = *(_QWORD *)(a4 + 32);
  if ( *(_QWORD *)(a4 + 24) == result )
    return sub_CB6200(a4, (unsigned __int8 *)v8, 1u);
  *(_BYTE *)result = *v8;
  ++*(_QWORD *)(a4 + 32);
  return result;
}
