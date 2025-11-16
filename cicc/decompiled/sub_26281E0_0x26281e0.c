// Function: sub_26281E0
// Address: 0x26281e0
//
__int64 __fastcall sub_26281E0(__int64 *a1, __int64 a2, int a3, unsigned __int64 *a4, _BYTE *a5)
{
  __int64 v5; // rax
  unsigned int v6; // r15d
  unsigned __int64 v11; // rsi
  unsigned __int64 v12; // rsi
  unsigned __int64 v13; // rax
  __int64 v14; // rbx
  __int64 result; // rax
  __int64 v16; // rdi
  int v17; // edx
  _BYTE *v18; // rax

  v5 = 1;
  v6 = 0;
  v11 = a1[3];
  do
  {
    if ( a1[v5 + 3] < v11 )
    {
      v11 = a1[v5 + 3];
      v6 = v5;
    }
    ++v5;
  }
  while ( v5 != 8 );
  *a4 = v11;
  v12 = (unsigned int)(a3 + v11);
  a1[v6 + 3] = v12;
  v13 = a1[1] - *a1;
  if ( v12 > v13 )
    sub_CD93F0(a1, v12 - v13);
  v14 = a2 + 8;
  result = (unsigned int)(1 << v6);
  *a5 = result;
  v16 = *(_QWORD *)(v14 + 16);
  v17 = 1 << v6;
  if ( v14 != v16 )
  {
    while ( 1 )
    {
      v18 = (_BYTE *)(*a1 + *a4 + *(_QWORD *)(v16 + 32));
      *v18 |= v17;
      result = sub_220EF30(v16);
      v16 = result;
      if ( v14 == result )
        break;
      LOBYTE(v17) = *a5;
    }
  }
  return result;
}
