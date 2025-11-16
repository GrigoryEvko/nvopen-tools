// Function: sub_25F6810
// Address: 0x25f6810
//
_DWORD *__fastcall sub_25F6810(_DWORD *a1, __int64 a2, unsigned int *a3)
{
  __int64 v3; // rsi
  _DWORD *v5; // r8
  __int64 v6; // rdx
  unsigned int v7; // esi
  __int64 v8; // rax
  unsigned int *v9; // rcx

  v3 = a2 - (_QWORD)a1;
  v5 = a1;
  v6 = 0x86BCA1AF286BCA1BLL * (v3 >> 3);
  if ( v3 > 0 )
  {
    v7 = *a3;
    do
    {
      while ( 1 )
      {
        v8 = v6 >> 1;
        v9 = &v5[38 * (v6 >> 1)];
        if ( v7 < *v9 )
          break;
        v5 = v9 + 38;
        v6 = v6 - v8 - 1;
        if ( v6 <= 0 )
          return v5;
      }
      v6 >>= 1;
    }
    while ( v8 > 0 );
  }
  return v5;
}
