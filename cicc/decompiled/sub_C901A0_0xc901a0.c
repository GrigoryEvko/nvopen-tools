// Function: sub_C901A0
// Address: 0xc901a0
//
__int64 __fastcall sub_C901A0(__int64 *a1, int a2)
{
  _QWORD *v2; // rax
  __int64 v4; // rsi
  _DWORD *v5; // r8
  unsigned int v6; // edi
  __int64 v7; // rdx
  __int64 v8; // rax
  _DWORD *v9; // rsi
  __int64 v10; // rdx
  unsigned int *v11; // rcx

  v2 = (_QWORD *)a1[1];
  v4 = *a1;
  if ( !v2 )
  {
    v2 = sub_C900E0(a1 + 1, v4);
    v4 = *a1;
  }
  v5 = (_DWORD *)*v2;
  v6 = a2 - *(_DWORD *)(v4 + 8);
  v7 = v2[1] - *v2;
  v8 = v7 >> 2;
  if ( v7 <= 0 )
    return 1;
  v9 = v5;
  do
  {
    while ( 1 )
    {
      v10 = v8 >> 1;
      v11 = &v9[v8 >> 1];
      if ( v6 <= *v11 )
        break;
      v9 = v11 + 1;
      v8 = v8 - v10 - 1;
      if ( v8 <= 0 )
        return (unsigned int)(v9 - v5) + 1;
    }
    v8 >>= 1;
  }
  while ( v10 > 0 );
  return (unsigned int)(v9 - v5) + 1;
}
