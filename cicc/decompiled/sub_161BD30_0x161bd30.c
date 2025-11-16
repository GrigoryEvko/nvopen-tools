// Function: sub_161BD30
// Address: 0x161bd30
//
__int64 __fastcall sub_161BD30(_QWORD *a1, unsigned int *a2, __int64 a3)
{
  unsigned int *v3; // r15
  unsigned int v4; // r13d
  _BYTE *v6; // rdi
  int v7; // ebx
  size_t v8; // rdx
  __int64 *v9; // r13
  __int64 v10; // r13
  __int64 v11; // rax
  __int64 v12; // rbx
  __int64 v13; // rsi
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 *v17; // r14
  __int64 v18; // r12
  __int64 v20; // [rsp+8h] [rbp-68h]
  _BYTE *v21; // [rsp+10h] [rbp-60h] BYREF
  __int64 v22; // [rsp+18h] [rbp-58h]
  _BYTE s[80]; // [rsp+20h] [rbp-50h] BYREF

  v3 = a2;
  v4 = a3 + 1;
  v6 = s;
  v7 = a3;
  v21 = s;
  v22 = 0x400000000LL;
  if ( (unsigned __int64)(a3 + 1) > 4 )
  {
    sub_16CD150(&v21, s, a3 + 1, 8);
    v6 = v21;
  }
  LODWORD(v22) = v4;
  v8 = 8LL * v4;
  v9 = (__int64 *)&v6[v8];
  if ( v6 != &v6[v8] )
  {
    memset(v6, 0, v8);
    v9 = (__int64 *)v21;
  }
  *v9 = sub_161BD10(a1, (__int64)"branch_weights", 14);
  v10 = sub_1643350(*a1);
  if ( v7 )
  {
    v11 = (unsigned int)(v7 - 1);
    v12 = 8;
    v20 = (__int64)&a2[v11 + 1];
    do
    {
      v13 = *v3++;
      v14 = sub_15A0680(v10, v13, 0);
      v17 = (__int64 *)&v21[v12];
      v12 += 8;
      *v17 = sub_161BD20((__int64)a1, v14, v15, v16);
    }
    while ( v3 != (unsigned int *)v20 );
  }
  v18 = sub_1627350(*a1, v21, (unsigned int)v22, 0, 1);
  if ( v21 != s )
    _libc_free((unsigned __int64)v21);
  return v18;
}
