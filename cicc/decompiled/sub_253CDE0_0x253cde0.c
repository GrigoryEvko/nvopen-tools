// Function: sub_253CDE0
// Address: 0x253cde0
//
__int64 __fastcall sub_253CDE0(__int64 a1, __int64 a2)
{
  _QWORD *v2; // rax
  _QWORD *v3; // rdx
  unsigned int v5; // r8d

  if ( *(_BYTE *)(a1 + 132) )
  {
    v2 = *(_QWORD **)(a1 + 112);
    v3 = &v2[*(unsigned int *)(a1 + 124)];
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
      return *(unsigned __int8 *)(a1 + 132);
    }
  }
  else
  {
    LOBYTE(v5) = sub_C8CA60(a1 + 104, a2) != 0;
    return v5;
  }
}
