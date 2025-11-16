// Function: sub_DEECD0
// Address: 0xdeecd0
//
__int64 __fastcall sub_DEECD0(
        __int64 a1,
        __int64 *a2,
        char *a3,
        __int64 a4,
        unsigned __int8 a5,
        unsigned __int8 a6,
        unsigned __int8 a7)
{
  unsigned int v8; // r13d
  bool v9; // cl
  __int64 *v10; // rax
  bool v11; // al
  __int64 v12; // r8
  __int64 v13; // r9
  __int64 v14; // rdx
  __int64 v15; // rcx
  __int64 v17; // r15
  __int64 v18; // rdx
  __int64 v19; // rcx
  __int64 v20; // r8
  __int64 v21; // r9
  __int64 v22; // [rsp-10h] [rbp-B0h]
  __int64 v23; // [rsp-8h] [rbp-A8h]
  __int64 *v24; // [rsp+0h] [rbp-A0h]
  unsigned __int64 v27; // [rsp+10h] [rbp-90h]
  bool v28; // [rsp+10h] [rbp-90h]
  __int64 v30; // [rsp+20h] [rbp-80h] BYREF
  __int64 v31; // [rsp+28h] [rbp-78h]
  __int64 v32; // [rsp+30h] [rbp-70h]
  char v33; // [rsp+38h] [rbp-68h]
  char *v34; // [rsp+40h] [rbp-60h] BYREF
  int v35; // [rsp+48h] [rbp-58h]
  char v36; // [rsp+50h] [rbp-50h] BYREF

  v8 = *(_WORD *)(a4 + 2) & 0x3F;
  v9 = (*(_BYTE *)(a4 + 1) & 2) != 0;
  if ( a5 )
  {
    v28 = v9;
    v8 = sub_B52870(v8);
    v27 = (unsigned __int64)v28 << 32;
  }
  else
  {
    v27 = (unsigned __int64)v9 << 32;
  }
  v24 = sub_DD8400((__int64)a2, *(_QWORD *)(a4 - 64));
  v10 = sub_DD8400((__int64)a2, *(_QWORD *)(a4 - 32));
  sub_DEE290((__int64)&v30, a2, a3, v27 | v8, (__int64 **)v24, (__int64 **)v10, a6, a7);
  v11 = sub_D96A50(v30);
  v14 = v22;
  v15 = v23;
  if ( v11 && sub_D96A50(v31) )
  {
    v17 = sub_DA7CB0((__int64)a2, (__int64)a3, a4, a5);
    if ( sub_D96A50(v17) )
    {
      sub_DA2D80(a1, a2, *(_BYTE **)(a4 - 64), *(_QWORD *)(a4 - 32), (__int64)a3, v8);
    }
    else
    {
      a2 = (__int64 *)v17;
      sub_D97F80(a1, v17, v18, v19, v20, v21);
    }
  }
  else
  {
    *(_QWORD *)a1 = v30;
    *(_QWORD *)(a1 + 8) = v31;
    *(_QWORD *)(a1 + 16) = v32;
    *(_BYTE *)(a1 + 24) = v33;
    *(_QWORD *)(a1 + 32) = a1 + 48;
    *(_QWORD *)(a1 + 40) = 0x400000000LL;
    if ( v35 )
    {
      a2 = (__int64 *)&v34;
      sub_D91460(a1 + 32, &v34, v14, v15, v12, v13);
    }
  }
  if ( v34 != &v36 )
    _libc_free(v34, a2);
  return a1;
}
