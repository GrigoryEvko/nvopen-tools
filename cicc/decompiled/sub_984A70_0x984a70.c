// Function: sub_984A70
// Address: 0x984a70
//
unsigned __int64 __fastcall sub_984A70(__int64 a1, __int64 a2)
{
  unsigned int v2; // edx
  __int64 v3; // rax
  unsigned __int64 result; // rax

  v2 = *(_DWORD *)(a1 + 8);
  if ( v2 > 0x40 )
    return sub_C47690(a1, a2);
  v3 = 0;
  if ( v2 != (_DWORD)a2 )
    v3 = *(_QWORD *)a1 << a2;
  result = (0xFFFFFFFFFFFFFFFFLL >> -(char)v2) & v3;
  if ( !v2 )
    result = 0;
  *(_QWORD *)a1 = result;
  return result;
}
