// Function: sub_C7F1D0
// Address: 0xc7f1d0
//
__int64 __fastcall sub_C7F1D0(__int64 a1, unsigned int a2, unsigned __int64 a3, int a4, char a5)
{
  _BYTE *v7; // rdi
  unsigned int v9; // edx
  unsigned __int64 v10; // r15
  unsigned __int64 v11; // r13
  _BYTE *v12; // rax
  _BYTE *v14; // rax
  _BYTE v15[48]; // [rsp+80h] [rbp-30h] BYREF
  __int64 savedregs; // [rsp+B0h] [rbp+0h] BYREF

  v7 = v15;
  do
  {
    *--v7 = a2 % 0xA + 48;
    v9 = a2;
    a2 /= 0xAu;
  }
  while ( v9 > 9 );
  v10 = (int)((unsigned int)&savedregs - 48 - (_DWORD)v7);
  if ( a5 )
  {
    v14 = *(_BYTE **)(a1 + 32);
    if ( (unsigned __int64)v14 >= *(_QWORD *)(a1 + 24) )
    {
      sub_CB5D20(a1, 45);
    }
    else
    {
      *(_QWORD *)(a1 + 32) = v14 + 1;
      *v14 = 45;
    }
  }
  if ( v10 >= a3 )
  {
    if ( a4 != 1 )
      return sub_CB6200(a1, &v15[-v10], v10);
  }
  else if ( a4 != 1 )
  {
    v11 = (int)((unsigned int)&savedregs - 48 - (_DWORD)v7);
    do
    {
      while ( 1 )
      {
        v12 = *(_BYTE **)(a1 + 32);
        if ( (unsigned __int64)v12 >= *(_QWORD *)(a1 + 24) )
          break;
        ++v11;
        *(_QWORD *)(a1 + 32) = v12 + 1;
        *v12 = 48;
        if ( a3 <= v11 )
          return sub_CB6200(a1, &v15[-v10], v10);
      }
      ++v11;
      sub_CB5D20(a1, 48);
    }
    while ( a3 > v11 );
    return sub_CB6200(a1, &v15[-v10], v10);
  }
  return sub_C7F120(a1, (__int64)&v15[-v10], v10);
}
