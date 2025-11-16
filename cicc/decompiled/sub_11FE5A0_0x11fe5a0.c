// Function: sub_11FE5A0
// Address: 0x11fe5a0
//
void __fastcall sub_11FE5A0(__int64 a1, __int64 a2, __int64 a3, __int64 *a4)
{
  __int64 v4; // rbp
  __int64 v6; // r8
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 v9; // rax
  int v10; // edx
  __int64 v11; // rax
  unsigned __int64 v12; // rsi
  const char *v13; // [rsp-38h] [rbp-38h] BYREF
  char v14; // [rsp-18h] [rbp-18h]
  char v15; // [rsp-17h] [rbp-17h]
  __int64 v16; // [rsp-8h] [rbp-8h]

  *a4 = 0;
  if ( a3 - a2 > 15 )
  {
    v6 = a2 + 16;
    v7 = 0;
    do
    {
      v8 = 16 * v7;
      ++a2;
      *a4 = v8;
      v7 = (unsigned int)(__int16)word_3F64060[*(unsigned __int8 *)(a2 - 1)] + v8;
      *a4 = v7;
    }
    while ( v6 != a2 );
  }
  a4[1] = 0;
  if ( a2 != a3 )
  {
    v9 = 0;
    v10 = 0;
    while ( 1 )
    {
      v11 = 16 * v9;
      ++v10;
      ++a2;
      a4[1] = v11;
      v9 = (unsigned int)(__int16)word_3F64060[*(unsigned __int8 *)(a2 - 1)] + v11;
      a4[1] = v9;
      if ( v10 > 15 )
        break;
      if ( a3 == a2 )
        return;
    }
    if ( a3 != a2 )
    {
      v16 = v4;
      v12 = *(_QWORD *)(a1 + 56);
      v15 = 1;
      v13 = "constant bigger than 128 bits detected";
      v14 = 3;
      sub_11FD800(a1, v12, (__int64)&v13, 2);
    }
  }
}
