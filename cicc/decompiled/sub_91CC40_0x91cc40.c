// Function: sub_91CC40
// Address: 0x91cc40
//
bool __fastcall sub_91CC40(__int64 a1, unsigned __int64 a2)
{
  _QWORD *v2; // rax
  _QWORD *v3; // r8
  _QWORD *v4; // rdi
  __int64 v5; // rcx
  __int64 v6; // rdx
  bool result; // al

  v2 = *(_QWORD **)(a1 + 40);
  v3 = (_QWORD *)(a1 + 32);
  if ( !v2 )
    return 0;
  v4 = (_QWORD *)(a1 + 32);
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
  result = 0;
  if ( v3 != v4 )
    return v4[4] <= a2;
  return result;
}
