// Function: sub_CB29A0
// Address: 0xcb29a0
//
unsigned __int64 __fastcall sub_CB29A0(_BYTE *a1, __int64 a2, __int64 a3)
{
  const char *v3; // r13
  size_t v5; // rax
  _DWORD *v6; // rcx
  size_t v7; // rdx
  unsigned __int64 result; // rax
  unsigned __int64 v9; // rdi
  char *v10; // rcx
  const char *v11; // r13
  unsigned int v12; // ecx
  unsigned int v13; // ecx
  __int64 v14; // rsi

  v3 = "true";
  if ( !*a1 )
    v3 = "false";
  v5 = strlen(v3);
  v6 = *(_DWORD **)(a3 + 32);
  v7 = v5;
  result = *(_QWORD *)(a3 + 24) - (_QWORD)v6;
  if ( v7 > result )
    return sub_CB6200(a3, v3, v7);
  if ( (unsigned int)v7 < 8 )
  {
    if ( (v7 & 4) != 0 )
    {
      *v6 = *(_DWORD *)v3;
      result = (unsigned int)v7;
      *(_DWORD *)((char *)v6 + (unsigned int)v7 - 4) = *(_DWORD *)&v3[(unsigned int)v7 - 4];
    }
    else if ( (_DWORD)v7 )
    {
      result = *(unsigned __int8 *)v3;
      *(_BYTE *)v6 = result;
      if ( (v7 & 2) != 0 )
      {
        result = (unsigned int)v7;
        *(_WORD *)((char *)v6 + (unsigned int)v7 - 2) = *(_WORD *)&v3[(unsigned int)v7 - 2];
      }
    }
  }
  else
  {
    v9 = (unsigned __int64)(v6 + 2) & 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)v6 = *(_QWORD *)v3;
    result = (unsigned int)v7;
    *(_QWORD *)((char *)v6 + (unsigned int)v7 - 8) = *(_QWORD *)&v3[(unsigned int)v7 - 8];
    v10 = (char *)v6 - v9;
    v11 = (const char *)(v3 - v10);
    v12 = (v7 + (_DWORD)v10) & 0xFFFFFFF8;
    if ( v12 >= 8 )
    {
      v13 = v12 & 0xFFFFFFF8;
      LODWORD(result) = 0;
      do
      {
        v14 = (unsigned int)result;
        result = (unsigned int)(result + 8);
        *(_QWORD *)(v9 + v14) = *(_QWORD *)&v11[v14];
      }
      while ( (unsigned int)result < v13 );
    }
  }
  *(_QWORD *)(a3 + 32) += v7;
  return result;
}
