// Function: sub_993A20
// Address: 0x993a20
//
__int64 __fastcall sub_993A20(__int64 a1)
{
  unsigned int *v1; // rax

  v1 = (unsigned int *)sub_C94E20(a1);
  if ( v1 )
    return *v1;
  else
    return *(unsigned int *)(a1 + 16);
}
