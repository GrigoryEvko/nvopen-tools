// Function: sub_9B5130
// Address: 0x9b5130
//
__int64 __fastcall sub_9B5130(__int64 a1, _BYTE *a2, __int64 a3, int a4, const __m128i *a5)
{
  __int64 v7; // rdi
  _BYTE *v11; // rdi
  unsigned int v12; // ebx
  _BYTE *v13; // rax

  if ( *a2 != 54 )
    return 0;
  if ( a1 != *((_QWORD *)a2 - 8) )
    return 0;
  v7 = *((_QWORD *)a2 - 4);
  if ( *(_BYTE *)v7 == 17 )
  {
    v11 = (_BYTE *)(v7 + 24);
  }
  else
  {
    if ( (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v7 + 8) + 8LL) - 17 > 1 )
      return 0;
    if ( *(_BYTE *)v7 > 0x15u )
      return 0;
    v13 = (_BYTE *)sub_AD7630(v7, 0);
    if ( !v13 || *v13 != 17 )
      return 0;
    v11 = v13 + 24;
  }
  if ( (a2[1] & 2) == 0 && ((a2[1] >> 1) & 2) == 0 )
    return 0;
  v12 = *((_DWORD *)v11 + 2);
  if ( v12 <= 0x40 )
  {
    if ( *(_QWORD *)v11 )
      return sub_9A6530(a1, a3, a5, a4 + 1);
    return 0;
  }
  if ( v12 == (unsigned int)sub_C444A0(v11) )
    return 0;
  return sub_9A6530(a1, a3, a5, a4 + 1);
}
