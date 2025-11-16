// Function: sub_E91E30
// Address: 0xe91e30
//
__int64 __fastcall sub_E91E30(_QWORD *a1, unsigned int a2, int a3)
{
  __int64 v4; // rdx
  __int16 *v5; // rax
  unsigned __int16 *v6; // rdi
  __int16 *v7; // rdx
  unsigned int v8; // esi
  int v10; // eax

  v4 = a1[1] + 24LL * a2;
  v5 = (__int16 *)(a1[7] + 2LL * *(unsigned int *)(v4 + 4));
  v6 = (unsigned __int16 *)(a1[11] + 2LL * *(unsigned int *)(v4 + 12));
  v7 = v5 + 1;
  LODWORD(v5) = *v5;
  v8 = a2 + (_DWORD)v5;
  if ( !(_WORD)v5 )
    return 0;
  if ( (unsigned __int16)v8 != a3 )
  {
    while ( 1 )
    {
      v10 = *v7++;
      if ( !(_WORD)v10 )
        break;
      v8 += v10;
      ++v6;
      if ( (unsigned __int16)v8 == a3 )
        return *v6;
    }
    return 0;
  }
  return *v6;
}
