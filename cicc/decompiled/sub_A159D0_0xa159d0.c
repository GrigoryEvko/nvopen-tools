// Function: sub_A159D0
// Address: 0xa159d0
//
__int64 __fastcall sub_A159D0(__int64 *a1, _QWORD *a2)
{
  __int64 v2; // rdi
  unsigned __int64 *v3; // rdx
  _QWORD *v5; // rax
  unsigned __int64 v6; // rsi
  _QWORD *v7; // r8
  _QWORD *v8; // rdi
  __int64 v9; // rcx
  __int64 v10; // rdx

  v2 = *a1;
  v3 = (unsigned __int64 *)(*a2 & 0xFFFFFFFFFFFFFFF8LL);
  if ( (*a2 & 1) != 0 && v3[1] )
    return sub_A3F3B0(v2 + 24);
  v5 = *(_QWORD **)(v2 + 608);
  v6 = *v3;
  v7 = (_QWORD *)(v2 + 600);
  if ( v5 )
  {
    v8 = (_QWORD *)(v2 + 600);
    do
    {
      while ( 1 )
      {
        v9 = v5[2];
        v10 = v5[3];
        if ( v6 <= v5[4] )
          break;
        v5 = (_QWORD *)v5[3];
        if ( !v10 )
          goto LABEL_9;
      }
      v8 = v5;
      v5 = (_QWORD *)v5[2];
    }
    while ( v9 );
LABEL_9:
    if ( v7 != v8 && v6 >= v8[4] )
      v7 = v8;
  }
  return *((unsigned int *)v7 + 10);
}
