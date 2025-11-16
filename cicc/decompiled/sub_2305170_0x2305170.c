// Function: sub_2305170
// Address: 0x2305170
//
void __fastcall sub_2305170(_QWORD *a1)
{
  unsigned __int64 v1; // r12

  v1 = a1[1];
  *a1 = &unk_4A0B010;
  if ( v1 )
  {
    sub_2E81F20(v1);
    j_j___libc_free_0(v1);
  }
}
