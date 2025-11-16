// Function: sub_38D7050
// Address: 0x38d7050
//
__int64 __fastcall sub_38D7050(_QWORD *a1, unsigned int a2, int a3)
{
  __int64 v3; // rax
  int v5; // r8d
  __int64 v6; // rax
  __int64 v7; // rdx
  unsigned __int16 *v8; // rax
  int v9; // ecx
  int v10; // ecx
  unsigned __int16 *v11; // rdi
  unsigned __int16 *v12; // rax
  int v13; // edx

  v3 = a2;
  v5 = v3;
  v6 = *a1 + 24 * v3;
  v7 = *(unsigned int *)(v6 + 12);
  v8 = (unsigned __int16 *)(a1[6] + 2LL * *(unsigned int *)(v6 + 4));
  v9 = *v8;
  if ( (_WORD)v9 )
  {
    v10 = v5 + v9;
    v11 = (unsigned __int16 *)(a1[10] + 2 * v7);
    v12 = v8 + 1;
    if ( a3 == (unsigned __int16)v10 )
      return *v11;
    while ( 1 )
    {
      v13 = *v12++;
      if ( !(_WORD)v13 )
        break;
      v10 += v13;
      ++v11;
      if ( a3 == (unsigned __int16)v10 )
        return *v11;
    }
  }
  return 0;
}
