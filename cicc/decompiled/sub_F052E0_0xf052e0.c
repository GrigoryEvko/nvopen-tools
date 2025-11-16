// Function: sub_F052E0
// Address: 0xf052e0
//
__int64 __fastcall sub_F052E0(void *a1)
{
  void *v1; // rax
  unsigned int v2; // eax

  v1 = (void *)sub_F05A00(a1);
  v2 = sub_F05240(v1);
  if ( v2 > 0x29 )
    BUG();
  return byte_3F888E0[v2];
}
