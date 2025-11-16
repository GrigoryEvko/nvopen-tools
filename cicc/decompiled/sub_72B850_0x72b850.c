// Function: sub_72B850
// Address: 0x72b850
//
void __fastcall sub_72B850(__int64 a1)
{
  __int64 v1; // rdx
  _QWORD *v2; // rax
  _QWORD *v3; // rdx

  if ( (*(_BYTE *)(a1 + 89) & 2) != 0 )
  {
    v1 = sub_72B840(*(_QWORD *)(a1 + 48));
    v2 = *(_QWORD **)(v1 + 224);
    if ( a1 == v2[3] )
    {
      *(_QWORD *)(v1 + 224) = *v2;
    }
    else
    {
      do
      {
        v3 = v2;
        v2 = (_QWORD *)*v2;
      }
      while ( a1 != v2[3] );
      *v3 = *v2;
    }
    *(_BYTE *)(a1 + 89) &= ~2u;
  }
}
