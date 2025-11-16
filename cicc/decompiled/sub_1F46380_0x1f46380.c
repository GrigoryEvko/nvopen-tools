// Function: sub_1F46380
// Address: 0x1f46380
//
__int64 __fastcall sub_1F46380(__int64 a1, void *a2)
{
  __int64 v2; // rax
  __int64 v3; // rax
  int v4; // edx
  unsigned int v5; // r8d

  v2 = sub_1F462F0(a1, (__int64)a2);
  v3 = sub_1F446D0(a2, v2);
  v5 = 1;
  if ( v3 )
  {
    LOBYTE(v5) = v3 != (_QWORD)a2;
    v5 |= v4;
  }
  return v5;
}
