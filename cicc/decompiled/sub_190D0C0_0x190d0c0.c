// Function: sub_190D0C0
// Address: 0x190d0c0
//
__int64 __fastcall sub_190D0C0(_QWORD *a1, unsigned __int64 *a2)
{
  __int64 v4; // rax
  unsigned __int64 v5; // rsi
  _QWORD *v6; // rdx
  _QWORD *v7; // r8
  __int64 v8; // r12
  unsigned __int64 v9; // rcx
  _QWORD *v10; // rax
  _BOOL8 v12; // rdi

  v4 = sub_22077B0(48);
  v5 = *a2;
  v6 = (_QWORD *)a1[2];
  v7 = a1 + 1;
  v8 = v4;
  *(_QWORD *)(v4 + 32) = *a2;
  *(_QWORD *)(v4 + 40) = a2[1];
  if ( v6 )
  {
    while ( 1 )
    {
      v9 = v6[4];
      v10 = (_QWORD *)v6[3];
      if ( v5 < v9 )
        v10 = (_QWORD *)v6[2];
      if ( !v10 )
        break;
      v6 = v10;
    }
    v12 = 1;
    if ( v7 != v6 )
      v12 = v9 > v5;
  }
  else
  {
    v6 = a1 + 1;
    v12 = 1;
  }
  sub_220F040(v12, v8, v6, a1 + 1);
  ++a1[5];
  return v8;
}
