// Function: sub_7E2C20
// Address: 0x7e2c20
//
__int64 __fastcall sub_7E2C20(__int64 a1)
{
  __int64 v1; // rax
  __int64 i; // r8

  v1 = *(_QWORD *)(a1 + 72);
  for ( i = 0; v1; v1 = *(_QWORD *)(v1 + 16) )
    i = v1;
  return i;
}
