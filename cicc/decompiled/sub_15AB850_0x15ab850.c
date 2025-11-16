// Function: sub_15AB850
// Address: 0x15ab850
//
void __fastcall sub_15AB850(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rbx

  if ( a3 )
  {
    v3 = a3;
    do
    {
      sub_15AB790(a1, *(unsigned __int8 **)(v3 - 8LL * *(unsigned int *)(v3 + 8)));
      if ( *(_DWORD *)(v3 + 8) != 2 )
        break;
      v3 = *(_QWORD *)(v3 - 8);
    }
    while ( v3 );
  }
}
