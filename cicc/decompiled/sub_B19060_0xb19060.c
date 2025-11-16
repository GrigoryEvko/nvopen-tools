// Function: sub_B19060
// Address: 0xb19060
//
__int64 __fastcall sub_B19060(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  _QWORD *v4; // rax
  _QWORD *v5; // rdx
  unsigned int v7; // r8d

  if ( *(_BYTE *)(a1 + 28) )
  {
    v4 = *(_QWORD **)(a1 + 8);
    v5 = &v4[*(unsigned int *)(a1 + 20)];
    if ( v4 == v5 )
    {
      return 0;
    }
    else
    {
      while ( *v4 != a2 )
      {
        if ( v5 == ++v4 )
          return 0;
      }
      return *(unsigned __int8 *)(a1 + 28);
    }
  }
  else
  {
    LOBYTE(v7) = sub_C8CA60(a1, a2, a3, a4) != 0;
    return v7;
  }
}
