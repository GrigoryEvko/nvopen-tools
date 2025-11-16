// Function: sub_CB6C70
// Address: 0xcb6c70
//
__int64 __fastcall sub_CB6C70(__int64 a1, unsigned int a2)
{
  unsigned int v2; // ebx
  unsigned int v3; // r12d

  v2 = a2;
  if ( a2 <= 0x4F )
    return sub_CB6200(a1, byte_3F6AB00, a2);
  do
  {
    v3 = 79;
    if ( v2 <= 0x4F )
      v3 = v2;
    sub_CB6200(a1, byte_3F6AB00, v3);
    v2 -= v3;
  }
  while ( v2 );
  return a1;
}
