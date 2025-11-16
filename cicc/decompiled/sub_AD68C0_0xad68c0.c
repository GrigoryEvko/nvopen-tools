// Function: sub_AD68C0
// Address: 0xad68c0
//
__int64 __fastcall sub_AD68C0(__int64 a1, unsigned int a2)
{
  __int64 v2; // rdx
  __int64 *v3; // rdi
  __int64 v4; // r12
  __int64 v5; // rdx
  __int64 v6; // rcx
  __int64 v7; // r8
  __int64 v8; // rax
  __int64 v10; // r13
  __int64 v11; // rax
  _QWORD *i; // rbx
  __int64 v13; // [rsp+0h] [rbp-40h] BYREF
  _QWORD *v14; // [rsp+8h] [rbp-38h]

  if ( *(_BYTE *)(sub_AC5230(a1) + 8)
    && *(_BYTE *)(sub_AC5230(a1) + 8) != 1
    && *(_BYTE *)(sub_AC5230(a1) + 8) != 2
    && *(_BYTE *)(sub_AC5230(a1) + 8) != 3 )
  {
    v10 = sub_AC5320(a1, a2);
    v11 = sub_AC5230(a1);
    return sub_AD64C0(v11, v10, 0);
  }
  sub_AC5470((__int64)&v13, a1, a2);
  v3 = (__int64 *)sub_BD5C60(a1, a1, v2);
  v4 = sub_AC8EA0(v3, &v13);
  v8 = sub_C33340(v3, &v13, v5, v6, v7);
  if ( v13 != v8 )
  {
    sub_C338F0(&v13);
    return v4;
  }
  if ( !v14 )
    return v4;
  for ( i = &v14[3 * *(v14 - 1)]; v14 != i; sub_91D830(i) )
    i -= 3;
  j_j_j___libc_free_0_0(i - 1);
  return v4;
}
