// Function: sub_117FBB0
// Address: 0x117fbb0
//
__int64 __fastcall sub_117FBB0(__int64 **a1, __int64 a2, __int64 a3)
{
  __int64 *v3; // r13
  unsigned int v4; // eax
  unsigned int v5; // r15d
  __int64 v6; // rcx
  unsigned int v7; // eax
  unsigned int v8; // r15d
  __int64 v9; // r14
  unsigned __int64 v10; // rdx
  unsigned int v11; // eax
  unsigned int v12; // r12d
  unsigned int v14; // eax
  __int64 v15; // [rsp+0h] [rbp-70h]
  const void **v17; // [rsp+10h] [rbp-60h] BYREF
  const void **v18; // [rsp+18h] [rbp-58h] BYREF
  __int64 v19; // [rsp+20h] [rbp-50h] BYREF
  unsigned int v20; // [rsp+28h] [rbp-48h]
  unsigned __int64 v21; // [rsp+30h] [rbp-40h] BYREF
  unsigned int v22; // [rsp+38h] [rbp-38h]

  v3 = *a1;
  v4 = sub_BCB060(**a1);
  v5 = v4 - 1;
  v20 = v4;
  v6 = 1LL << ((unsigned __int8)v4 - 1);
  if ( v4 <= 0x40 )
  {
    v19 = 0;
LABEL_3:
    v19 |= v6;
    goto LABEL_4;
  }
  v15 = 1LL << ((unsigned __int8)v4 - 1);
  sub_C43690((__int64)&v19, 0, 0);
  v6 = v15;
  if ( v20 <= 0x40 )
  {
    v3 = *a1;
    goto LABEL_3;
  }
  *(_QWORD *)(v19 + 8LL * (v5 >> 6)) |= v15;
  v3 = *a1;
LABEL_4:
  v7 = sub_BCB060(*v3);
  v8 = v7 - 1;
  v22 = v7;
  v9 = ~(1LL << ((unsigned __int8)v7 - 1));
  if ( v7 <= 0x40 )
  {
    v10 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v7;
    if ( !v7 )
      v10 = 0;
    v21 = v10;
    goto LABEL_8;
  }
  sub_C43690((__int64)&v21, -1, 1);
  if ( v22 <= 0x40 )
  {
LABEL_8:
    v21 &= v9;
    goto LABEL_9;
  }
  *(_QWORD *)(v21 + 8LL * (v8 >> 6)) &= v9;
LABEL_9:
  v17 = (const void **)&v19;
  LOBYTE(v11) = sub_10080A0(&v17, a2);
  v12 = v11;
  if ( (_BYTE)v11 )
  {
    v18 = (const void **)&v21;
    LOBYTE(v14) = sub_10080A0(&v18, a3);
    v12 = v14;
  }
  if ( v22 > 0x40 && v21 )
    j_j___libc_free_0_0(v21);
  if ( v20 > 0x40 && v19 )
    j_j___libc_free_0_0(v19);
  return v12;
}
