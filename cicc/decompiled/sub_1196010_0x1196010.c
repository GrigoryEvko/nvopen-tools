// Function: sub_1196010
// Address: 0x1196010
//
bool __fastcall sub_1196010(_QWORD *a1, __int64 a2)
{
  _BYTE *v2; // r12
  unsigned int v3; // ebx
  __int64 v4; // rax
  __int64 v6; // rdx
  _BYTE *v7; // rax

  v2 = (_BYTE *)a2;
  if ( *(_BYTE *)a2 != 17 )
  {
    v6 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(a2 + 8) + 8LL) - 17;
    if ( (unsigned int)v6 > 1 )
      return 0;
    if ( *(_BYTE *)a2 > 0x15u )
      return 0;
    v7 = sub_AD7630(a2, 1, v6);
    v2 = v7;
    if ( !v7 || *v7 != 17 )
      return 0;
  }
  v3 = *((_DWORD *)v2 + 8);
  if ( v3 <= 0x40 )
  {
    v4 = *((_QWORD *)v2 + 3);
    return *a1 == v4;
  }
  if ( v3 - (unsigned int)sub_C444A0((__int64)(v2 + 24)) <= 0x40 )
  {
    v4 = **((_QWORD **)v2 + 3);
    return *a1 == v4;
  }
  return 0;
}
