// Function: sub_2671260
// Address: 0x2671260
//
__int64 __fastcall sub_2671260(__int64 a1, __int64 a2)
{
  unsigned int v2; // r8d
  _QWORD *v4; // rax
  _QWORD *v5; // rdx
  unsigned int v6; // r8d

  v2 = *(unsigned __int8 *)(a1 + 97);
  if ( !(_BYTE)v2 )
    return v2;
  v2 = *(unsigned __int8 *)(a1 + 500);
  if ( (_BYTE)v2 )
  {
    v4 = *(_QWORD **)(a1 + 480);
    v5 = &v4[*(unsigned int *)(a1 + 492)];
    if ( v4 == v5 )
      return v2;
    while ( a2 != *v4 )
    {
      if ( v5 == ++v4 )
        return v2;
    }
    return 0;
  }
  else
  {
    LOBYTE(v6) = sub_C8CA60(a1 + 472, a2) == 0;
    return v6;
  }
}
