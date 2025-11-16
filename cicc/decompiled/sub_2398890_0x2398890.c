// Function: sub_2398890
// Address: 0x2398890
//
_QWORD *__fastcall sub_2398890(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rax
  _QWORD *v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // r8
  __int64 v12; // r9
  _QWORD *v13; // rbx
  __int64 v14; // rdi
  __int64 v15; // rax
  __int64 v17; // [rsp+8h] [rbp-98h] BYREF
  __int64 v18; // [rsp+10h] [rbp-90h]
  __int64 v19; // [rsp+18h] [rbp-88h]
  unsigned int v20; // [rsp+20h] [rbp-80h]
  __int64 v21; // [rsp+28h] [rbp-78h]
  __int64 v22; // [rsp+40h] [rbp-60h]
  __int64 v23; // [rsp+48h] [rbp-58h] BYREF
  __int64 v24; // [rsp+50h] [rbp-50h]
  __int64 v25; // [rsp+58h] [rbp-48h]
  unsigned int v26; // [rsp+60h] [rbp-40h]
  __int64 v27; // [rsp+68h] [rbp-38h]

  v7 = *(_QWORD *)(a2 + 8);
  v17 = 0;
  v18 = 1;
  v22 = v7;
  v19 = -4096;
  v21 = -4096;
  v23 = 0;
  v24 = 1;
  v25 = -4096;
  v27 = -4096;
  sub_2398550((__int64)&v23, (__int64)&v17, a3, a4, a5, a6);
  v8 = (_QWORD *)sub_22077B0(0x40u);
  v13 = v8;
  if ( v8 )
  {
    v14 = (__int64)(v8 + 2);
    v8[2] = 0;
    v8[3] = 1;
    v8[4] = -4096;
    *v8 = &unk_4A0AB38;
    v15 = v22;
    v13[6] = -4096;
    v13[1] = v15;
    sub_2398550(v14, (__int64)&v23, v9, v10, v11, v12);
  }
  sub_2398130((__int64)&v23);
  if ( (v24 & 1) == 0 )
    sub_C7D6A0(v25, 16LL * v26, 8);
  *a1 = v13;
  sub_2398130((__int64)&v17);
  if ( (v18 & 1) == 0 )
    sub_C7D6A0(v19, 16LL * v20, 8);
  return a1;
}
