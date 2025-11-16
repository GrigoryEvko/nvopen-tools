// Function: sub_1DB4CE0
// Address: 0x1db4ce0
//
void __fastcall sub_1DB4CE0(__int64 a1)
{
  unsigned __int64 *v1; // rbx
  unsigned __int64 *v2; // rsi

  v1 = *(unsigned __int64 **)(a1 + 104);
  while ( v1 )
  {
    v2 = v1;
    v1 = (unsigned __int64 *)v1[13];
    sub_1DB4BE0(a1, v2);
  }
  *(_QWORD *)(a1 + 104) = 0;
}
