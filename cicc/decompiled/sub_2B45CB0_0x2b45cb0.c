// Function: sub_2B45CB0
// Address: 0x2b45cb0
//
unsigned __int64 __fastcall sub_2B45CB0(__int64 *a1, __int64 a2)
{
  unsigned __int8 *v3; // r14
  unsigned int v4; // eax
  __int64 v5; // rsi
  unsigned int v6; // r8d
  unsigned int v7; // edx
  __int64 v8; // rax
  __int64 v9; // rax
  bool v10; // of
  unsigned __int64 v11; // rax
  unsigned __int64 v12; // r12
  bool v14; // cc
  __int64 v15; // [rsp+0h] [rbp-80h] BYREF
  int v16; // [rsp+8h] [rbp-78h]
  __int64 v17; // [rsp+10h] [rbp-70h]
  int v18; // [rsp+18h] [rbp-68h]
  char *v19; // [rsp+20h] [rbp-60h] BYREF
  unsigned int v20; // [rsp+28h] [rbp-58h]
  char v21; // [rsp+30h] [rbp-50h] BYREF

  v3 = (unsigned __int8 *)*a1;
  v4 = sub_9B78C0(*a1, *(__int64 **)(a1[1] + 3304));
  v5 = a1[5];
  v6 = 0;
  v7 = v4;
  v8 = a1[1];
  if ( v5 != *(_QWORD *)(v8 + 3528) + 24LL * *(unsigned int *)(v8 + 3544) )
    v6 = *(_DWORD *)(v5 + 8);
  sub_2B1F480((__int64)&v19, v3, v7, *(_DWORD *)(a1[2] + 32), v6, *(_QWORD *)(v8 + 3296));
  sub_2B45470(
    (unsigned int *)&v15,
    (__int64)v3,
    a1[2],
    *(_QWORD *)(a1[1] + 3296),
    *(__int64 **)(a1[1] + 3304),
    a1[2],
    v19,
    v20);
  if ( v18 == v16 )
  {
    v9 = v15;
    if ( v17 < v15 )
      v9 = v17;
  }
  else if ( v18 < v16 )
  {
    v9 = v17;
  }
  else
  {
    v9 = v15;
  }
  v10 = __OFADD__(a2, v9);
  v11 = a2 + v9;
  if ( v10 )
  {
    v14 = a2 <= 0;
    v12 = 0x7FFFFFFFFFFFFFFFLL;
    if ( v14 )
      v12 = 0x8000000000000000LL;
  }
  else
  {
    v12 = v11;
  }
  if ( v19 != &v21 )
    _libc_free((unsigned __int64)v19);
  return v12;
}
