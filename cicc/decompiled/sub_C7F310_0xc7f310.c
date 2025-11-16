// Function: sub_C7F310
// Address: 0xc7f310
//
__int64 __fastcall sub_C7F310(__int64 a1, unsigned __int64 a2, unsigned __int64 a3, int a4, char a5)
{
  _BYTE *v8; // rdi
  unsigned __int64 v9; // rax
  unsigned __int64 v10; // rbx
  _BYTE *v11; // r15
  unsigned __int64 v12; // r14
  _BYTE *v13; // rax
  _BYTE *v15; // rax
  _BYTE v16[48]; // [rsp+80h] [rbp-30h] BYREF
  __int64 savedregs; // [rsp+B0h] [rbp+0h] BYREF

  v8 = v16;
  if ( a2 == (unsigned int)a2 )
    return sub_C7F1D0(a1, a2, a3, a4, a5);
  do
  {
    *--v8 = a2 % 0xA + 48;
    v9 = a2;
    a2 /= 0xAu;
  }
  while ( v9 > 9 );
  v10 = (int)((unsigned int)&savedregs - 48 - (_DWORD)v8);
  if ( !a5 )
  {
LABEL_4:
    v11 = &v16[-v10];
    if ( a3 > v10 )
      goto LABEL_5;
LABEL_15:
    if ( a4 != 1 )
      return sub_CB6200(a1, v11, v10);
    return sub_C7F120(a1, (__int64)v11, v10);
  }
  v15 = *(_BYTE **)(a1 + 32);
  if ( (unsigned __int64)v15 >= *(_QWORD *)(a1 + 24) )
  {
    sub_CB5D20(a1, 45);
    goto LABEL_4;
  }
  *(_QWORD *)(a1 + 32) = v15 + 1;
  v11 = &v16[-v10];
  *v15 = 45;
  if ( a3 <= v10 )
    goto LABEL_15;
LABEL_5:
  if ( a4 != 1 )
  {
    v12 = (int)((unsigned int)&savedregs - 48 - (_DWORD)v8);
    do
    {
      while ( 1 )
      {
        v13 = *(_BYTE **)(a1 + 32);
        if ( (unsigned __int64)v13 >= *(_QWORD *)(a1 + 24) )
          break;
        ++v12;
        *(_QWORD *)(a1 + 32) = v13 + 1;
        *v13 = 48;
        if ( a3 <= v12 )
          return sub_CB6200(a1, v11, v10);
      }
      ++v12;
      sub_CB5D20(a1, 48);
    }
    while ( a3 > v12 );
    return sub_CB6200(a1, v11, v10);
  }
  return sub_C7F120(a1, (__int64)v11, v10);
}
