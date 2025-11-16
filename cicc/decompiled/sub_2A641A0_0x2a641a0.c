// Function: sub_2A641A0
// Address: 0x2a641a0
//
__int64 __fastcall sub_2A641A0(__int64 *a1, __int64 a2)
{
  __int64 v2; // rdi
  _QWORD *v3; // rax
  _QWORD *v4; // rdx
  unsigned int v6; // r8d

  v2 = *a1;
  if ( *(_BYTE *)(v2 + 708) )
  {
    v3 = *(_QWORD **)(v2 + 688);
    v4 = &v3[*(unsigned int *)(v2 + 700)];
    if ( v3 == v4 )
    {
      return 0;
    }
    else
    {
      while ( a2 != *v3 )
      {
        if ( v4 == ++v3 )
          return 0;
      }
      return *(unsigned __int8 *)(v2 + 708);
    }
  }
  else
  {
    LOBYTE(v6) = sub_C8CA60(v2 + 680, a2) != 0;
    return v6;
  }
}
