// Function: sub_16F4D80
// Address: 0x16f4d80
//
__int64 __fastcall sub_16F4D80(__int64 a1, unsigned __int64 a2, size_t a3, int a4, char a5)
{
  _BYTE *v8; // rdi
  unsigned __int64 v9; // rax
  size_t v10; // rbx
  char *v11; // r15
  size_t v12; // r14
  _BYTE *v13; // rax
  _BYTE *v15; // rax
  __m128i si128; // [rsp+70h] [rbp-40h]
  _BYTE v17[48]; // [rsp+80h] [rbp-30h] BYREF
  __int64 savedregs; // [rsp+B0h] [rbp+0h] BYREF

  if ( a2 == (unsigned int)a2 )
    return sub_16F4C10(a1, a2, a3, a4, a5);
  v8 = v17;
  si128 = _mm_load_si128((const __m128i *)&xmmword_3F66F70);
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
LABEL_5:
    v11 = &v17[-v10];
    if ( a3 > v10 )
      goto LABEL_6;
LABEL_16:
    if ( a4 != 1 )
      return sub_16E7EE0(a1, v11, v10);
    return sub_16F4B60(a1, v11, v10);
  }
  v15 = *(_BYTE **)(a1 + 24);
  if ( (unsigned __int64)v15 >= *(_QWORD *)(a1 + 16) )
  {
    sub_16E7DE0(a1, 45);
    goto LABEL_5;
  }
  *(_QWORD *)(a1 + 24) = v15 + 1;
  v11 = &v17[-v10];
  *v15 = 45;
  if ( a3 <= v10 )
    goto LABEL_16;
LABEL_6:
  if ( a4 != 1 )
  {
    v12 = (int)((unsigned int)&savedregs - 48 - (_DWORD)v8);
    do
    {
      while ( 1 )
      {
        v13 = *(_BYTE **)(a1 + 24);
        if ( (unsigned __int64)v13 >= *(_QWORD *)(a1 + 16) )
          break;
        ++v12;
        *(_QWORD *)(a1 + 24) = v13 + 1;
        *v13 = 48;
        if ( a3 <= v12 )
          return sub_16E7EE0(a1, v11, v10);
      }
      ++v12;
      sub_16E7DE0(a1, 48);
    }
    while ( a3 > v12 );
    return sub_16E7EE0(a1, v11, v10);
  }
  return sub_16F4B60(a1, v11, v10);
}
