// Function: sub_B9D9A0
// Address: 0xb9d9a0
//
__int64 __fastcall sub_B9D9A0(__int64 *a1, __int64 *a2, __int64 *a3)
{
  _BYTE *v4; // r13
  __int64 *v6; // rax
  unsigned int v7; // r14d
  __int64 v8; // [rsp+18h] [rbp-38h]

  if ( !a3 )
    return sub_B9C770(a1, a2, a3, 0, 1);
  v4 = (_BYTE *)*a2;
  if ( !*a2 || (unsigned __int8)(*v4 - 5) > 0x1Fu )
    return sub_B9C770(a1, a2, a3, 0, 1);
  v6 = (*(v4 - 16) & 2) != 0 ? (__int64 *)*((unsigned int *)v4 - 6) : (__int64 *)((*((_WORD *)v4 - 8) >> 6) & 0xF);
  if ( v6 != a3 || v4 != *(_BYTE **)sub_A17150(v4 - 16) )
    return sub_B9C770(a1, a2, a3, 0, 1);
  if ( a3 != (__int64 *)1 )
  {
    v7 = 1;
    while ( 1 )
    {
      v8 = a2[v7];
      if ( v8 != *(_QWORD *)&sub_A17150(v4 - 16)[8 * v7] )
        break;
      if ( (_DWORD)a3 == ++v7 )
        return (__int64)v4;
    }
    return sub_B9C770(a1, a2, a3, 0, 1);
  }
  return (__int64)v4;
}
