// Function: sub_C656D0
// Address: 0xc656d0
//
__int64 __fastcall sub_C656D0(__int64 a1, char a2)
{
  __int64 v2; // rbx
  __int64 result; // rax
  __int64 v4; // rdx
  __int64 v5; // rcx
  __int64 v6; // r8
  __int64 v7; // r9

  v2 = (unsigned int)(1 << a2);
  *(_DWORD *)(a1 + 8) = v2;
  result = _libc_calloc((unsigned int)(v2 + 1), 8);
  if ( !result )
  {
    if ( (_DWORD)v2 != -1 )
      sub_C64F00("Allocation failed", 1u);
    result = sub_C65340(1, 8, v4, v5, v6, v7);
  }
  *(_QWORD *)(result + 8 * v2) = -1;
  *(_QWORD *)a1 = result;
  *(_DWORD *)(a1 + 12) = 0;
  return result;
}
