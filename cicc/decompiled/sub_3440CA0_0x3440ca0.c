// Function: sub_3440CA0
// Address: 0x3440ca0
//
unsigned __int64 __fastcall sub_3440CA0(__int64 a1, unsigned int a2)
{
  unsigned int v2; // edx
  __int64 v3; // rax
  unsigned __int64 result; // rax

  v2 = *(_DWORD *)(a1 + 8);
  if ( v2 > 0x40 )
    return sub_C47690((__int64 *)a1, a2);
  v3 = 0;
  if ( v2 != a2 )
    v3 = *(_QWORD *)a1 << a2;
  result = (0xFFFFFFFFFFFFFFFFLL >> -(char)v2) & v3;
  if ( !v2 )
    result = 0;
  *(_QWORD *)a1 = result;
  return result;
}
