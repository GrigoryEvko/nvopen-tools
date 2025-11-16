// Function: sub_1E31380
// Address: 0x1e31380
//
void __fastcall sub_1E31380(__int64 a1)
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
          v3 = *(_QWORD *)(v2 + 56);
          if ( v3 )
            sub_1E69A50(*(_QWORD *)(v3 + 40), a1);
        }
      }
    }
  }
}
