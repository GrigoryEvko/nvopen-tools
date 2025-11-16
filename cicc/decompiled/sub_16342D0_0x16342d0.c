// Function: sub_16342D0
// Address: 0x16342d0
//
__int64 __fastcall sub_16342D0(__int64 a1, unsigned __int64 a2)
{
  _QWORD *v2; // rax
  _QWORD *v4; // r9
  _QWORD *v5; // rdi
  __int64 v6; // rcx
  __int64 v7; // rdx
  unsigned __int64 v8; // rcx

  v2 = *(_QWORD **)(a1 + 16);
  v4 = (_QWORD *)(a1 + 8);
  if ( v2 )
  {
    v5 = (_QWORD *)(a1 + 8);
    do
    {
      while ( 1 )
      {
        v6 = v2[2];
        v7 = v2[3];
        if ( a2 <= v2[4] )
          break;
        v2 = (_QWORD *)v2[3];
        if ( !v7 )
          goto LABEL_6;
      }
      v5 = v2;
      v2 = (_QWORD *)v2[2];
    }
    while ( v6 );
LABEL_6:
    v8 = 0;
    if ( v4 != v5 && a2 >= v5[4] )
      v8 = (unsigned __int64)(v5 + 4) & 0xFFFFFFFFFFFFFFFBLL;
  }
  else
  {
    v8 = 0;
  }
  return **(_QWORD **)(((v8 | (4LL * *(unsigned __int8 *)(a1 + 178))) & 0xFFFFFFFFFFFFFFF8LL) + 0x18);
}
