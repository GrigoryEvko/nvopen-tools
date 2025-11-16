// Function: sub_1E404B0
// Address: 0x1e404b0
//
__int64 __fastcall sub_1E404B0(__int64 a1, unsigned __int64 a2)
{
  _QWORD *v2; // rax
  _QWORD *v3; // r8
  __int64 v4; // rcx
  __int64 v5; // rdx
  __int64 result; // rax

  v2 = *(_QWORD **)(a1 + 48);
  if ( !v2 )
    return 0xFFFFFFFFLL;
  v3 = (_QWORD *)(a1 + 40);
  do
  {
    while ( 1 )
    {
      v4 = v2[2];
      v5 = v2[3];
      if ( v2[4] >= a2 )
        break;
      v2 = (_QWORD *)v2[3];
      if ( !v5 )
        goto LABEL_6;
    }
    v3 = v2;
    v2 = (_QWORD *)v2[2];
  }
  while ( v4 );
LABEL_6:
  result = 0xFFFFFFFFLL;
  if ( (_QWORD *)(a1 + 40) != v3 && v3[4] <= a2 )
    return (unsigned int)((*((_DWORD *)v3 + 10) - *(_DWORD *)(a1 + 128)) / *(_DWORD *)(a1 + 136));
  return result;
}
