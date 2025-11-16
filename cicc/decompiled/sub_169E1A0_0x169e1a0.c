// Function: sub_169E1A0
// Address: 0x169e1a0
//
__int64 __fastcall sub_169E1A0(__int64 a1, __int64 a2, unsigned int a3, _BYTE *a4)
{
  _BYTE *v6; // rdi
  __int64 v7; // rax
  unsigned int v8; // r14d
  unsigned __int64 v9; // rax
  unsigned __int64 v10; // rdx
  unsigned int v11; // r12d
  _BYTE *v12; // rdi
  __int64 v14; // [rsp+0h] [rbp-90h]
  unsigned __int8 v15; // [rsp+10h] [rbp-80h]
  unsigned __int64 v16; // [rsp+10h] [rbp-80h]
  __int64 v18; // [rsp+20h] [rbp-70h] BYREF
  int v19; // [rsp+28h] [rbp-68h]
  _BYTE *v20; // [rsp+30h] [rbp-60h] BYREF
  __int64 v21; // [rsp+38h] [rbp-58h]
  _BYTE s[80]; // [rsp+40h] [rbp-50h] BYREF

  v6 = s;
  v7 = *(unsigned int *)(a2 + 8);
  v20 = s;
  v8 = v7;
  v9 = v7 + 63;
  v21 = 0x400000000LL;
  v10 = v9 >> 6;
  if ( v9 > 0x13F )
  {
    v16 = v9 >> 6;
    sub_16CD150(&v20, s, v10, 8);
    v6 = v20;
    v10 = v16;
  }
  LODWORD(v21) = v10;
  if ( 8 * v10 )
  {
    memset(v6, 0, 8 * v10);
    v6 = v20;
    v10 = (unsigned int)v21;
  }
  v14 = v10;
  v15 = *(_BYTE *)(a2 + 12) ^ 1;
  if ( *(void **)(a1 + 8) == sub_16982C0() )
    v11 = sub_169E030(a1 + 8, v6, v14, v8, v15, a3, a4);
  else
    v11 = sub_169A0A0(a1 + 8, v6, v14, v8, v15, a3, a4);
  sub_16A50F0(&v18, v8, v20, (unsigned int)v21);
  if ( *(_DWORD *)(a2 + 8) > 0x40u && *(_QWORD *)a2 )
    j_j___libc_free_0_0(*(_QWORD *)a2);
  v12 = v20;
  *(_QWORD *)a2 = v18;
  *(_DWORD *)(a2 + 8) = v19;
  if ( v12 != s )
    _libc_free((unsigned __int64)v12);
  return v11;
}
