// Function: sub_1A9BBE0
// Address: 0x1a9bbe0
//
_QWORD *__fastcall sub_1A9BBE0(__int64 *a1, __int64 *a2, __int64 a3)
{
  _QWORD *v4; // r12
  __int64 v5; // r14
  _QWORD *v6; // rax
  _QWORD *v7; // rbx
  _QWORD *v9; // [rsp+8h] [rbp-48h] BYREF
  char *v10; // [rsp+10h] [rbp-40h] BYREF
  char v11; // [rsp+20h] [rbp-30h]
  char v12; // [rsp+21h] [rbp-2Fh]

  v9 = (_QWORD *)sub_1A9B680((__int64)a2, *a1);
  if ( sub_1A94B30((__int64)v9) )
    v4 = v9;
  else
    v4 = *(_QWORD **)(sub_1A9BA30(a1[1], (__int64 *)&v9) + 8);
  v5 = *a2;
  if ( *v4 != *a2 && a3 )
  {
    v12 = 1;
    v10 = "cast";
    v11 = 3;
    v6 = sub_1648A60(56, 1u);
    v7 = v6;
    if ( v6 )
      sub_15FD590((__int64)v6, (__int64)v4, v5, (__int64)&v10, a3);
    return v7;
  }
  return v4;
}
