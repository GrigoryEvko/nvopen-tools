// Function: sub_AE2980
// Address: 0xae2980
//
_DWORD *__fastcall sub_AE2980(__int64 a1, unsigned int a2)
{
  __int64 v2; // r8
  __int64 v4; // rax
  _DWORD *v5; // rdi
  _DWORD *v6; // r9
  __int64 i; // rax
  unsigned int *v8; // rcx

  v2 = *(_QWORD *)(a1 + 272);
  if ( !a2 )
    return (_DWORD *)v2;
  v4 = *(unsigned int *)(a1 + 280);
  v5 = *(_DWORD **)(a1 + 272);
  v4 *= 20;
  v6 = (_DWORD *)(v2 + v4);
  for ( i = 0xCCCCCCCCCCCCCCCDLL * (v4 >> 2); i > 0; i >>= 1 )
  {
    while ( 1 )
    {
      v8 = &v5[5 * (i >> 1)];
      if ( *v8 >= a2 )
        break;
      v5 = v8 + 5;
      i = i - (i >> 1) - 1;
      if ( i <= 0 )
        goto LABEL_7;
    }
  }
LABEL_7:
  if ( v6 == v5 )
    return (_DWORD *)v2;
  if ( *v5 == a2 )
    return v5;
  return (_DWORD *)v2;
}
