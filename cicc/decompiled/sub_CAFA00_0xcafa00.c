// Function: sub_CAFA00
// Address: 0xcafa00
//
__int64 __fastcall sub_CAFA00(__int64 **a1, _BYTE *a2, __int64 a3, __int64 a4, __int64 a5)
{
  unsigned int i; // r14d
  __int64 v6; // rax
  int v7; // edx
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // r8
  int v12; // [rsp+0h] [rbp-60h]
  _QWORD *v13; // [rsp+18h] [rbp-48h] BYREF
  _QWORD v14[7]; // [rsp+28h] [rbp-38h] BYREF

  for ( i = 0; ; i = 1 )
  {
    v6 = sub_CAD7A0(a1, (unsigned __int64)a2, a3, a4, a5);
    v7 = *(_DWORD *)v6;
    v13 = v14;
    a2 = *(_BYTE **)(v6 + 24);
    v12 = v7;
    sub_CA64F0((__int64 *)&v13, a2, (__int64)&a2[*(_QWORD *)(v6 + 32)]);
    if ( v12 != 4 )
      break;
    sub_CAF680((__int64)a1, (__int64)a2, v8, v9, v10);
LABEL_4:
    if ( v13 != v14 )
    {
      a2 = (_BYTE *)(v14[0] + 1LL);
      j_j___libc_free_0(v13, v14[0] + 1LL);
    }
  }
  if ( v12 == 3 )
  {
    sub_CAD6E0((unsigned __int64 **)a1, (__int64)a2, v8, v9, v10);
    goto LABEL_4;
  }
  if ( v13 != v14 )
    j_j___libc_free_0(v13, v14[0] + 1LL);
  return i;
}
