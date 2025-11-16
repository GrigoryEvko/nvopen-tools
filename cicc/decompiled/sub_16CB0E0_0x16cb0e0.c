// Function: sub_16CB0E0
// Address: 0x16cb0e0
//
__int64 __fastcall sub_16CB0E0(__int64 a1, char *a2, __int64 a3)
{
  char *v3; // r13
  char *v4; // rbx
  char v5; // si
  __int64 result; // rax

  v3 = &a2[a3];
  if ( a2 != &a2[a3] )
  {
    v4 = a2;
    do
    {
      v5 = *v4++;
      result = sub_16CB0D0(a1, v5);
    }
    while ( v3 != v4 );
  }
  return result;
}
