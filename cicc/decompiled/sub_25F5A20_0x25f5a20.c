// Function: sub_25F5A20
// Address: 0x25f5a20
//
_QWORD *__fastcall sub_25F5A20(__int64 *a1, __int64 a2)
{
  __int64 v2; // r12
  __int64 v3; // rax
  unsigned int v4; // r14d
  __int64 v5; // r13
  int v6; // r12d
  __int64 *v7; // rax

  v2 = a2;
  v3 = sub_B92180(*a1);
  if ( v3 && a2 && *(_BYTE *)a2 == 6 )
  {
    v4 = *(unsigned __int16 *)(a2 + 2);
    v5 = v3;
    v6 = *(_DWORD *)(a2 + 4);
    v7 = (__int64 *)sub_B2BE50(*a1);
    return sub_B01860(v7, v6, v4, v5, 0, 0, 0, 1);
  }
  return (_QWORD *)v2;
}
