// Function: sub_A72A00
// Address: 0xa72a00
//
__int64 __fastcall sub_A72A00(__int64 a1)
{
  _DWORD *v1; // rax
  unsigned int v2; // r8d
  __int64 v3; // rdx

  v1 = (_DWORD *)sub_A72230(a1);
  v2 = 0;
  if ( v3 == 4 )
    LOBYTE(v2) = *v1 == 1702195828;
  return v2;
}
