// Function: sub_10E4480
// Address: 0x10e4480
//
bool __fastcall sub_10E4480(_QWORD *a1, __int64 a2, __int64 a3)
{
  _BYTE *v3; // r12
  unsigned int v4; // ebx
  __int64 v5; // rax
  _BYTE *v7; // rax

  v3 = (_BYTE *)a2;
  if ( *(_BYTE *)a2 != 17 )
  {
    if ( (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(a2 + 8) + 8LL) - 17 > 1 )
      return 0;
    v7 = sub_AD7630(a2, 0, a3);
    v3 = v7;
    if ( !v7 || *v7 != 17 )
      return 0;
  }
  v4 = *((_DWORD *)v3 + 8);
  if ( v4 <= 0x40 )
  {
    v5 = *((_QWORD *)v3 + 3);
    return *a1 == v5;
  }
  if ( v4 - (unsigned int)sub_C444A0((__int64)(v3 + 24)) <= 0x40 )
  {
    v5 = **((_QWORD **)v3 + 3);
    return *a1 == v5;
  }
  return 0;
}
