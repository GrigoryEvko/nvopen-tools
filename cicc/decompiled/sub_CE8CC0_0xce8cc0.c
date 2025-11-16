// Function: sub_CE8CC0
// Address: 0xce8cc0
//
__int64 __fastcall sub_CE8CC0(__int64 a1)
{
  int v1; // ebx
  char v2; // r8
  char v3; // al
  __int64 v5; // [rsp+8h] [rbp-18h] BYREF

  v2 = sub_CE7ED0(a1, "reqntidx", 8u, &v5);
  v3 = 0;
  if ( v2 )
  {
    v1 = v5;
    v3 = 1;
  }
  LODWORD(v5) = v1;
  BYTE4(v5) = v3;
  return v5;
}
