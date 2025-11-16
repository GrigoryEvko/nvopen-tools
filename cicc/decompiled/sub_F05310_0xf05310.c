// Function: sub_F05310
// Address: 0xf05310
//
__int64 __fastcall sub_F05310(void *a1)
{
  void *v1; // rax
  unsigned int v2; // eax

  v1 = (void *)sub_F05A00(a1);
  v2 = sub_F05240(v1);
  if ( v2 > 0x29 )
    BUG();
  return (unsigned int)byte_3F88920[v2];
}
