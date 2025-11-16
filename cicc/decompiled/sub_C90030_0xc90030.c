// Function: sub_C90030
// Address: 0xc90030
//
__int64 __fastcall sub_C90030(__int64 *a1, __int16 a2)
{
  _QWORD *v2; // rax
  __int64 v4; // rsi
  _WORD *v5; // r8
  unsigned __int16 v6; // di
  __int64 v7; // rdx
  __int64 v8; // rax
  _WORD *v9; // rcx
  __int64 v10; // rsi
  unsigned __int16 *v11; // rdx

  v2 = (_QWORD *)a1[1];
  v4 = *a1;
  if ( !v2 )
  {
    v2 = sub_C8FF60(a1 + 1, v4);
    v4 = *a1;
  }
  v5 = (_WORD *)*v2;
  v6 = a2 - *(_WORD *)(v4 + 8);
  v7 = v2[1] - *v2;
  v8 = v7 >> 1;
  if ( v7 <= 0 )
    return 1;
  v9 = v5;
  do
  {
    while ( 1 )
    {
      v10 = v8 >> 1;
      v11 = (_WORD *)((char *)v9 + (v8 & 0xFFFFFFFFFFFFFFFELL));
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
