// Function: sub_321E150
// Address: 0x321e150
//
__int64 __fastcall sub_321E150(_QWORD *a1, __int64 *a2)
{
  _QWORD *v2; // rax
  __int64 v3; // rdx
  __int64 v4; // rcx
  __int64 v5; // r8
  __int64 v6; // r9
  _QWORD *v7; // rbx
  __int64 result; // rax

  v2 = (_QWORD *)sub_22077B0(0x50u);
  v7 = v2;
  if ( v2 )
  {
    *v2 = *a2;
    v2[1] = v2 + 3;
    v2[2] = 0x200000000LL;
    if ( *((_DWORD *)a2 + 4) )
      sub_3218BB0((__int64)(v2 + 1), (__int64)(a2 + 1), v3, v4, v5, v6);
    *((_BYTE *)v7 + 72) = *((_BYTE *)a2 + 72);
  }
  *a1 = v7;
  result = *a2;
  if ( !(unsigned int)((__int64)(*(_QWORD *)(*a2 + 24) - *(_QWORD *)(*a2 + 16)) >> 3) )
    result = 0;
  a1[1] = result;
  return result;
}
