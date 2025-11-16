// Function: sub_153E7E0
// Address: 0x153e7e0
//
__int64 __fastcall sub_153E7E0(__int64 a1, unsigned __int64 a2)
{
  _QWORD *v2; // rax
  _QWORD *v3; // r8
  _QWORD *v4; // rdi
  __int64 v5; // rcx
  __int64 v6; // rdx
  __int64 result; // rax

  v2 = *(_QWORD **)(a1 + 152);
  v3 = (_QWORD *)(a1 + 144);
  if ( !v2 )
    return 0;
  v4 = (_QWORD *)(a1 + 144);
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
  if ( v3 != v4 && v4[4] <= a2 )
    return *((unsigned int *)v4 + 10);
  return result;
}
