// Function: sub_E91D60
// Address: 0xe91d60
//
__int64 __fastcall sub_E91D60(_QWORD *a1, unsigned int a2, int a3, __int64 a4)
{
  __int16 *v6; // rax
  __int16 *v7; // r13
  int v8; // eax
  int v9; // r15d
  unsigned __int16 i; // dx
  unsigned int v11; // ebx
  __int64 v12; // rax
  int v13; // eax
  int v14; // eax

  v6 = (__int16 *)(a1[7] + 2LL * *(unsigned int *)(a1[1] + 24LL * a2 + 8));
  v7 = v6 + 1;
  v8 = *v6;
  v9 = a2 + v8;
  if ( !(_WORD)v8 )
    return 0;
  for ( i = a2 + v8; ; i = v9 )
  {
    v11 = i;
    v12 = i >> 3;
    if ( (unsigned int)v12 < *(unsigned __int16 *)(a4 + 22) )
    {
      v13 = *(unsigned __int8 *)(*(_QWORD *)(a4 + 8) + v12);
      if ( _bittest(&v13, i & 7) )
      {
        if ( a2 == (unsigned int)sub_E91CF0(a1, i, a3) )
          break;
      }
    }
    v14 = *v7++;
    if ( !(_WORD)v14 )
      return 0;
    v9 += v14;
  }
  return v11;
}
