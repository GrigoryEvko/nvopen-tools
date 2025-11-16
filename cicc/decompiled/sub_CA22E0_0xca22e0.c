// Function: sub_CA22E0
// Address: 0xca22e0
//
void __fastcall sub_CA22E0(volatile signed __int32 **a1)
{
  volatile signed __int32 *v1; // rdi

  v1 = *a1;
  if ( v1 )
  {
    if ( !_InterlockedSub(v1 + 2, 1u) )
      (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v1 + 8LL))(v1);
  }
}
