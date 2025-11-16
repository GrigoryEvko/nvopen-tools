// Function: sub_CA1930
// Address: 0xca1930
//
__int64 __fastcall sub_CA1930(_BYTE *a1)
{
  if ( a1[8] )
    sub_CA17B0("Cannot implicitly convert a scalable size to a fixed-width size in `TypeSize::operator ScalarTy()`");
  return *(_QWORD *)a1;
}
