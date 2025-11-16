// Function: sub_8D72A0
// Address: 0x8d72a0
//
__int64 __fastcall sub_8D72A0(__int64 a1)
{
  __int64 v1; // r12
  __int64 v3; // rbx
  __int64 i; // rax

  v1 = *(_QWORD *)(a1 + 8);
  if ( dword_4F077C4 == 2 )
  {
    if ( dword_4F07734 )
    {
      v3 = *(_QWORD *)(a1 + 16);
      if ( v3 )
      {
        if ( sub_8D3410(*(_QWORD *)(a1 + 16)) )
        {
          for ( i = v3; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
            ;
          if ( *(char *)(i + 168) < 0 )
            return v3;
        }
      }
    }
  }
  return v1;
}
