// Function: sub_1CB7740
// Address: 0x1cb7740
//
__int64 __fastcall sub_1CB7740(unsigned int *a1, __int64 a2)
{
  int v3; // ebx
  unsigned int v4; // esi
  __int64 result; // rax

  v3 = 1 << (*(unsigned __int16 *)(a2 + 18) >> 1);
  v4 = sub_1CB76C0(a1, *(_QWORD *)(a2 - 24));
  result = 0;
  if ( v4 > v3 >> 1 && *a1 != v4 && a1[1] != v4 )
  {
    sub_15F9450(a2, v4);
    return 1;
  }
  return result;
}
