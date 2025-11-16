// Function: sub_38D6F10
// Address: 0x38d6f10
//
__int64 __fastcall sub_38D6F10(_QWORD *a1, unsigned int a2, int a3)
{
  __int64 v5; // r9
  __int64 v6; // rdx
  _WORD *v7; // rdi
  unsigned __int16 *v8; // r8
  unsigned __int16 v9; // cx
  __int64 v10; // rax
  __int16 v11; // dx

  v5 = a1[10];
  v6 = *a1 + 24LL * a2;
  v7 = (_WORD *)(a1[6] + 2LL * *(unsigned int *)(v6 + 4));
  if ( *v7 )
  {
    v8 = (unsigned __int16 *)(v5 + 2LL * *(unsigned int *)(v6 + 12));
    v9 = a2 + *v7;
    v10 = 0;
    if ( *v8 == a3 )
      return v9;
    while ( 1 )
    {
      v11 = v7[++v10];
      if ( !v11 )
        break;
      v9 += v11;
      if ( v8[v10] == a3 )
        return v9;
    }
  }
  return 0;
}
