// Function: sub_31A6B20
// Address: 0x31a6b20
//
__int64 __fastcall sub_31A6B20(__int64 a1, _BYTE *a2)
{
  unsigned int v2; // r8d
  _QWORD *v3; // rax
  _QWORD *v4; // rdx
  unsigned int v6; // r8d

  v2 = 0;
  if ( *a2 <= 0x1Cu )
    return v2;
  v2 = *(unsigned __int8 *)(a1 + 204);
  if ( !(_BYTE)v2 )
  {
    LOBYTE(v6) = sub_C8CA60(a1 + 176, (__int64)a2) != 0;
    return v6;
  }
  v3 = *(_QWORD **)(a1 + 184);
  v4 = &v3[*(unsigned int *)(a1 + 196)];
  if ( v3 != v4 )
  {
    while ( a2 != (_BYTE *)*v3 )
    {
      if ( v4 == ++v3 )
        return 0;
    }
    return v2;
  }
  return 0;
}
