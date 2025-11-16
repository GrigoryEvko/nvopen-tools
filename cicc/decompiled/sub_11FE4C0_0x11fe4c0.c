// Function: sub_11FE4C0
// Address: 0x11fe4c0
//
void __fastcall sub_11FE4C0(__int64 a1, __int64 a2, __int64 a3, __int64 *a4)
{
  __int64 v4; // rbp
  __int64 v6; // rax
  int v8; // edx
  __int64 v9; // rax
  __int64 v10; // rax
  int v11; // edx
  __int64 v12; // rax
  unsigned __int64 v13; // rsi
  const char *v14; // [rsp-38h] [rbp-38h] BYREF
  char v15; // [rsp-18h] [rbp-18h]
  char v16; // [rsp-17h] [rbp-17h]
  __int64 v17; // [rsp-8h] [rbp-8h]

  a4[1] = 0;
  if ( a2 == a3 )
  {
    *a4 = 0;
  }
  else
  {
    v6 = 0;
    v8 = 0;
    do
    {
      v9 = 16 * v6;
      ++v8;
      ++a2;
      a4[1] = v9;
      v6 = (unsigned int)(__int16)word_3F64060[*(unsigned __int8 *)(a2 - 1)] + v9;
      a4[1] = v6;
    }
    while ( v8 <= 3 && a3 != a2 );
    *a4 = 0;
    if ( a3 != a2 )
    {
      v10 = 0;
      v11 = 0;
      while ( 1 )
      {
        v12 = 16 * v10;
        ++v11;
        ++a2;
        *a4 = v12;
        v10 = (unsigned int)(__int16)word_3F64060[*(unsigned __int8 *)(a2 - 1)] + v12;
        *a4 = v10;
        if ( v11 > 15 )
          break;
        if ( a3 == a2 )
          return;
      }
      if ( a3 != a2 )
      {
        v17 = v4;
        v13 = *(_QWORD *)(a1 + 56);
        v16 = 1;
        v14 = "constant bigger than 128 bits detected";
        v15 = 3;
        sub_11FD800(a1, v13, (__int64)&v14, 2);
      }
    }
  }
}
