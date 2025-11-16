// Function: sub_16F4C10
// Address: 0x16f4c10
//
__int64 __fastcall sub_16F4C10(__int64 a1, unsigned int a2, size_t a3, int a4, char a5)
{
  _BYTE *v7; // rdi
  unsigned int v9; // edx
  size_t v10; // r15
  size_t v11; // r13
  _BYTE *v12; // rax
  _BYTE *v14; // rax
  __m128i si128; // [rsp+70h] [rbp-40h]
  _BYTE v16[48]; // [rsp+80h] [rbp-30h] BYREF
  __int64 savedregs; // [rsp+B0h] [rbp+0h] BYREF

  v7 = v16;
  si128 = _mm_load_si128((const __m128i *)&xmmword_3F66F70);
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
    v14 = *(_BYTE **)(a1 + 24);
    if ( (unsigned __int64)v14 >= *(_QWORD *)(a1 + 16) )
    {
      sub_16E7DE0(a1, 45);
    }
    else
    {
      *(_QWORD *)(a1 + 24) = v14 + 1;
      *v14 = 45;
    }
  }
  if ( v10 >= a3 )
  {
    if ( a4 != 1 )
      return sub_16E7EE0(a1, &v16[-v10], v10);
  }
  else if ( a4 != 1 )
  {
    v11 = (int)((unsigned int)&savedregs - 48 - (_DWORD)v7);
    do
    {
      while ( 1 )
      {
        v12 = *(_BYTE **)(a1 + 24);
        if ( (unsigned __int64)v12 >= *(_QWORD *)(a1 + 16) )
          break;
        ++v11;
        *(_QWORD *)(a1 + 24) = v12 + 1;
        *v12 = 48;
        if ( a3 <= v11 )
          return sub_16E7EE0(a1, &v16[-v10], v10);
      }
      ++v11;
      sub_16E7DE0(a1, 48);
    }
    while ( a3 > v11 );
    return sub_16E7EE0(a1, &v16[-v10], v10);
  }
  return sub_16F4B60(a1, &v16[-v10], v10);
}
