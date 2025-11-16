// Function: sub_E31D10
// Address: 0xe31d10
//
void __fastcall sub_E31D10(__int64 a1, unsigned __int64 a2)
{
  _BYTE *v3; // r12
  unsigned __int64 v4; // rax
  _BYTE *v5; // r8
  size_t v6; // r13
  __int64 v7; // rdx
  unsigned __int64 v8; // rax
  char *v9; // rdi
  unsigned __int64 v10; // rsi
  unsigned __int64 v11; // rax
  __int64 v12; // rax
  _BYTE v13[51]; // [rsp-33h] [rbp-33h] BYREF

  if ( !*(_BYTE *)(a1 + 49) )
  {
    if ( *(_BYTE *)(a1 + 48) )
    {
      v3 = v13;
      do
      {
        *--v3 = a2 % 0xA + 48;
        v4 = a2;
        a2 /= 0xAu;
      }
      while ( v4 > 9 );
      v5 = (_BYTE *)(v13 - v3);
      v6 = v13 - v3;
      if ( v13 != v3 )
      {
        v7 = *(_QWORD *)(a1 + 64);
        v8 = *(_QWORD *)(a1 + 72);
        v9 = *(char **)(a1 + 56);
        if ( (unsigned __int64)&v5[v7] > v8 )
        {
          v10 = (unsigned __int64)&v5[v7 + 992];
          v11 = 2 * v8;
          if ( v10 > v11 )
            *(_QWORD *)(a1 + 72) = v10;
          else
            *(_QWORD *)(a1 + 72) = v11;
          v12 = realloc(v9);
          *(_QWORD *)(a1 + 56) = v12;
          v9 = (char *)v12;
          if ( !v12 )
            abort();
          v7 = *(_QWORD *)(a1 + 64);
        }
        memcpy(&v9[v7], v3, v6);
        *(_QWORD *)(a1 + 64) += v6;
      }
    }
  }
}
