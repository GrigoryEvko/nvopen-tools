// Function: sub_127C950
// Address: 0x127c950
//
__int64 __fastcall sub_127C950(__int64 a1, unsigned __int64 a2)
{
  _QWORD *v2; // rax
  _QWORD *v3; // r8
  _QWORD *v4; // rdi
  __int64 v5; // rcx
  __int64 v6; // rdx

  v2 = *(_QWORD **)(a1 + 88);
  v3 = (_QWORD *)(a1 + 80);
  if ( v2 )
  {
    v4 = (_QWORD *)(a1 + 80);
    do
    {
      while ( 1 )
      {
        v5 = v2[2];
        v6 = v2[3];
        if ( v2[4] >= a2 )
          break;
        v2 = (_QWORD *)v2[3];
        if ( !v6 )
          goto LABEL_6;
      }
      v4 = v2;
      v2 = (_QWORD *)v2[2];
    }
    while ( v5 );
LABEL_6:
    if ( v3 != v4 && v4[4] <= a2 )
      v3 = v4;
  }
  return v3[5];
}
