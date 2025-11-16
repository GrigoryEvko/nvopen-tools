// Function: sub_2FF0C20
// Address: 0x2ff0c20
//
__int64 __fastcall sub_2FF0C20(__int64 a1, void *a2)
{
  __int64 v2; // rax
  __int64 v3; // rax
  int v4; // edx
  unsigned int v5; // r8d

  v2 = sub_2FF0B90(a1, (__int64)a2);
  v3 = sub_2FEDBC0(a2, v2);
  v5 = 1;
  if ( v3 )
  {
    LOBYTE(v5) = v3 != (_QWORD)a2;
    v5 |= v4;
  }
  return v5;
}
