// Function: sub_C33A50
// Address: 0xc33a50
//
__int64 __fastcall sub_C33A50(__int64 a1)
{
  _QWORD *v1; // rax
  unsigned int v2; // r8d
  __int64 v3; // rdx
  _QWORD *v4; // rbx
  unsigned int v5; // eax
  unsigned int v6; // eax
  __int64 v7; // rcx
  __int64 *v8; // rax
  __int64 v9; // rdx

  v1 = (_QWORD *)sub_C33930(a1);
  v2 = 0;
  v3 = *v1;
  if ( (*v1 & 1) != 0 )
    return v2;
  v4 = v1;
  v5 = (unsigned int)(*(_DWORD *)(*(_QWORD *)a1 + 8LL) + 63) >> 6;
  if ( !v5 )
    v5 = 1;
  v6 = v5 - 1;
  v7 = v6;
  if ( !v6 )
  {
LABEL_10:
    LOBYTE(v2) = (*v4 | (-1LL << (64 - (unsigned __int8)sub_C337A0(a1))) | 1) == -1;
    return v2;
  }
  if ( (~(_DWORD)v3 & 0xFFFFFFFE) != 0 )
    return v2;
  v8 = v4 + 1;
  do
  {
    if ( &v4[v7] == v8 )
    {
      v4 += v7;
      goto LABEL_10;
    }
    v9 = *v8++;
  }
  while ( (_DWORD)v9 == -1 );
  return 0;
}
