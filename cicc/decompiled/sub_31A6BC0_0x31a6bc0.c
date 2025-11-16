// Function: sub_31A6BC0
// Address: 0x31a6bc0
//
__int64 __fastcall sub_31A6BC0(__int64 a1, __int64 a2)
{
  _QWORD *v2; // rax
  _QWORD *v3; // rdx
  unsigned int v5; // r8d

  if ( *(_BYTE *)(a1 + 268) )
  {
    v2 = *(_QWORD **)(a1 + 248);
    v3 = &v2[*(unsigned int *)(a1 + 260)];
    if ( v2 == v3 )
    {
      return 0;
    }
    else
    {
      while ( a2 != *v2 )
      {
        if ( v3 == ++v2 )
          return 0;
      }
      return *(unsigned __int8 *)(a1 + 268);
    }
  }
  else
  {
    LOBYTE(v5) = sub_C8CA60(a1 + 240, a2) != 0;
    return v5;
  }
}
