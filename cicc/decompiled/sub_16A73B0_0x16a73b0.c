// Function: sub_16A73B0
// Address: 0x16a73b0
//
__int64 __fastcall sub_16A73B0(_QWORD *a1, __int64 a2, int a3)
{
  bool v3; // cf
  __int64 *v4; // rdi
  __int64 *v5; // rdx
  __int64 v6; // rax

  if ( !a3 )
    return 1;
  v3 = __CFADD__(*a1, a2);
  *a1 += a2;
  if ( v3 )
  {
    v4 = a1 + 1;
    v5 = &v4[a3 - 1];
    while ( v4 != v5 )
    {
      v6 = *v4++;
      *(v4 - 1) = ++v6;
      if ( v6 )
        return 0;
    }
    return 1;
  }
  return 0;
}
