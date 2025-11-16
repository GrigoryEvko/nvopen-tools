// Function: sub_1AB1BA0
// Address: 0x1ab1ba0
//
__int64 __fastcall sub_1AB1BA0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 *a5, int a6)
{
  __int64 v8; // r13
  __int64 *v9; // rdi
  __int64 v10; // rax
  __int64 v11; // r13
  __int64 v12; // r12
  __int64 v13; // rax
  __int64 v15; // [rsp+0h] [rbp-B0h]
  __int64 v16; // [rsp+8h] [rbp-A8h]
  __int64 v17; // [rsp+10h] [rbp-A0h] BYREF
  __int64 v18; // [rsp+18h] [rbp-98h]
  _QWORD *v19; // [rsp+20h] [rbp-90h]
  __int64 **v20; // [rsp+28h] [rbp-88h] BYREF
  unsigned __int64 v21[2]; // [rsp+30h] [rbp-80h] BYREF
  _QWORD v22[2]; // [rsp+40h] [rbp-70h] BYREF
  unsigned __int64 v23[2]; // [rsp+50h] [rbp-60h] BYREF
  _BYTE v24[80]; // [rsp+60h] [rbp-50h] BYREF

  v17 = a2;
  v18 = a3;
  v23[1] = 0x1400000000LL;
  v20 = (__int64 **)a1;
  v23[0] = (unsigned __int64)v24;
  if ( *(_BYTE *)(*(_QWORD *)a1 + 8LL) != 3 )
    sub_1AAE630(a1, (__int64)&v17, (__int64)v23, a4, (int)a5, a6);
  v8 = sub_157EB90(*(_QWORD *)(a4 + 8));
  v19 = v22;
  v9 = *v20;
  v15 = v18;
  v21[0] = (unsigned __int64)v22;
  v16 = v17;
  v22[0] = v9;
  v21[1] = 0x100000001LL;
  v10 = sub_1644EA0(v9, v22, 1, 0);
  v11 = sub_1632080(v8, v16, v15, v10, 0);
  if ( (_QWORD *)v21[0] != v19 )
    _libc_free(v21[0]);
  v21[0] = (unsigned __int64)&v17;
  LOWORD(v22[0]) = 261;
  v12 = sub_1285290((__int64 *)a4, *(_QWORD *)(*(_QWORD *)v11 + 24LL), v11, (int)&v20, 1, (__int64)v21, 0);
  *(_QWORD *)(v12 + 56) = sub_1563C10(a5, *(__int64 **)(a4 + 24), -1, 47);
  v13 = sub_1649C60(v11);
  if ( !*(_BYTE *)(v13 + 16) )
    *(_WORD *)(v12 + 18) = *(_WORD *)(v12 + 18) & 0x8000
                         | *(_WORD *)(v12 + 18) & 3
                         | (*(_WORD *)(v13 + 18) >> 2) & 0xFFC;
  if ( (_BYTE *)v23[0] != v24 )
    _libc_free(v23[0]);
  return v12;
}
