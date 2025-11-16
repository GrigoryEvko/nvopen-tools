// Function: sub_C8FD40
// Address: 0xc8fd40
//
__int64 __fastcall sub_C8FD40(__int64 *a1, char a2)
{
  _QWORD *v2; // rax
  __int64 v4; // rsi
  _BYTE *v5; // r8
  unsigned __int8 v6; // di
  __int64 v7; // rax
  _BYTE *v8; // rsi
  __int64 v9; // rdx
  unsigned __int8 *v10; // rcx

  v2 = (_QWORD *)a1[1];
  v4 = *a1;
  if ( !v2 )
  {
    v2 = sub_C8FC80(a1 + 1, v4);
    v4 = *a1;
  }
  v5 = (_BYTE *)*v2;
  v6 = a2 - *(_BYTE *)(v4 + 8);
  v7 = v2[1] - *v2;
  if ( v7 <= 0 )
    return 1;
  v8 = v5;
  do
  {
    while ( 1 )
    {
      v9 = v7 >> 1;
      v10 = &v8[v7 >> 1];
      if ( v6 <= *v10 )
        break;
      v8 = v10 + 1;
      v7 = v7 - v9 - 1;
      if ( v7 <= 0 )
        return (unsigned int)((_DWORD)v8 - (_DWORD)v5 + 1);
    }
    v7 >>= 1;
  }
  while ( v9 > 0 );
  return (unsigned int)((_DWORD)v8 - (_DWORD)v5 + 1);
}
