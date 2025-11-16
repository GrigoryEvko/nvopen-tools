// Function: sub_35459E0
// Address: 0x35459e0
//
__int64 __fastcall sub_35459E0(__int64 a1, __int64 a2, _QWORD *a3)
{
  __int64 v4; // rax
  __int64 v5; // r10
  __int64 v6; // r8
  unsigned __int64 v7; // rsi
  _QWORD *v8; // rdi
  _QWORD *v9; // rax
  __int64 v10; // rcx
  __int64 v11; // rdx

  v4 = sub_35459D0(a3, a2);
  v5 = *(_QWORD *)v4;
  v6 = *(_QWORD *)v4 + 32LL * *(unsigned int *)(v4 + 8);
  if ( v6 == *(_QWORD *)v4 )
    return 1;
  while ( 1 )
  {
    v7 = *(_QWORD *)(v5 + 8) & 0xFFFFFFFFFFFFFFF8LL;
    if ( *(_QWORD *)(a1 + 48) )
    {
      v8 = (_QWORD *)(a1 + 40);
      v9 = *(_QWORD **)(a1 + 48);
      do
      {
        while ( 1 )
        {
          v10 = v9[2];
          v11 = v9[3];
          if ( v9[4] >= v7 )
            break;
          v9 = (_QWORD *)v9[3];
          if ( !v11 )
            goto LABEL_7;
        }
        v8 = v9;
        v9 = (_QWORD *)v9[2];
      }
      while ( v10 );
LABEL_7:
      if ( (_QWORD *)(a1 + 40) != v8 && v8[4] <= v7 )
        return 0;
    }
    v5 += 32;
    if ( v5 == v6 )
      return 1;
  }
}
