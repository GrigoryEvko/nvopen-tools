// Function: sub_2EAB370
// Address: 0x2eab370
//
void __fastcall sub_2EAB370(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // rax
  __int64 v3; // rax

  if ( !*(_BYTE *)a1 )
  {
    if ( *(_QWORD *)(a1 + 24) )
    {
      v1 = *(_QWORD *)(a1 + 16);
      if ( v1 )
      {
        v2 = *(_QWORD *)(v1 + 24);
        if ( v2 )
        {
          v3 = *(_QWORD *)(v2 + 32);
          if ( v3 )
            sub_2EBEB60(*(_QWORD *)(v3 + 32), a1);
        }
      }
    }
  }
}
