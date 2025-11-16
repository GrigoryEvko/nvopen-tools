// Function: sub_1E6E610
// Address: 0x1e6e610
//
__int64 __fastcall sub_1E6E610(__int64 *a1, const char *a2, const char *a3, __int64 a4)
{
  size_t v4; // rax
  size_t v7; // rax

  v4 = 0;
  *a1 = 0;
  a1[1] = (__int64)a2;
  if ( a2 )
    v4 = strlen(a2);
  a1[2] = v4;
  v7 = 0;
  a1[3] = (__int64)a3;
  if ( a3 )
    v7 = strlen(a3);
  a1[5] = a4;
  a1[4] = v7;
  return sub_1E40390((__int64 *)&unk_4FC7850, a1);
}
