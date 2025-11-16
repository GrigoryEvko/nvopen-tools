// Function: sub_77F710
// Address: 0x77f710
//
__int64 __fastcall sub_77F710(__int64 a1, int a2)
{
  _QWORD *v2; // rax
  _QWORD *v3; // rbx
  _QWORD *v4; // rdx
  unsigned __int8 v5; // dl
  _BYTE *v6; // rax

  v2 = *(_QWORD **)(a1 + 200);
  v3 = (_QWORD *)(a1 + 200);
  if ( !v2 )
  {
LABEL_8:
    v6 = sub_724980();
    *v3 = v6;
    if ( a2 )
      v6[8] |= 1u;
    else
      v6[8] |= 2u;
    *(_QWORD *)(*v3 + 16LL) = 0;
    return *v3;
  }
  v4 = (_QWORD *)*v2;
  if ( *v2 )
  {
    do
    {
      v3 = v2;
      v2 = v4;
      v4 = (_QWORD *)*v4;
    }
    while ( v4 );
  }
  v5 = *((_BYTE *)v2 + 8);
  if ( !a2 )
    v5 >>= 1;
  if ( (v5 & 1) == 0 )
  {
    v3 = v2;
    goto LABEL_8;
  }
  return *v3;
}
