// Function: sub_256B410
// Address: 0x256b410
//
__int64 __fastcall sub_256B410(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  unsigned __int64 v3; // r13
  __int64 *v4; // rax
  __int64 v5; // r8
  char v6; // al
  unsigned int v7; // r14d
  __int64 v8; // rax
  _QWORD *v9; // r12
  _QWORD *v10; // r13
  unsigned __int64 v11; // rbx
  unsigned __int64 v12; // rdi
  unsigned __int64 v13; // rdi
  __int64 v15; // [rsp+8h] [rbp-C8h]
  int v16; // [rsp+24h] [rbp-ACh] BYREF
  __int64 *v17; // [rsp+28h] [rbp-A8h] BYREF
  __int64 v18[2]; // [rsp+30h] [rbp-A0h] BYREF
  __int64 v19; // [rsp+40h] [rbp-90h] BYREF
  _QWORD *v20; // [rsp+48h] [rbp-88h]
  __int64 v21; // [rsp+50h] [rbp-80h]
  unsigned int v22; // [rsp+58h] [rbp-78h]
  _QWORD v23[14]; // [rsp+60h] [rbp-70h] BYREF

  v2 = *(_QWORD *)(a2 + 208);
  v16 = 1;
  v15 = *(_QWORD *)(v2 + 104);
  v19 = 0;
  v3 = sub_250D070((_QWORD *)(a1 + 72));
  v21 = 0;
  v20 = 0;
  v22 = 0;
  v17 = (__int64 *)v3;
  v4 = sub_256A430((__int64)&v19, (__int64 *)&v17);
  v18[0] = 0;
  sub_256AFA0((__int64)v23, (__int64)v4, v18, (__int64)v18, v5);
  v23[5] = &v16;
  v23[3] = v15;
  v17 = &v19;
  v23[1] = &v19;
  v18[0] = (__int64)&v19;
  v23[0] = &v17;
  v23[2] = a2;
  v23[4] = a1;
  v23[6] = v3;
  v18[1] = (__int64)&v17;
  v6 = sub_252FFB0(
         a2,
         (unsigned __int8 (__fastcall *)(__int64, __int64, __int64 *))sub_2584500,
         (__int64)v23,
         a1,
         v3,
         1,
         1,
         1,
         (unsigned __int8 (__fastcall *)(__int64, __int64, __int64))sub_256B2D0,
         (__int64)v18);
  v7 = v16;
  if ( !v6 )
  {
    v7 = 0;
    *(_BYTE *)(a1 + 393) = *(_BYTE *)(a1 + 392);
  }
  v8 = v22;
  if ( v22 )
  {
    v9 = v20;
    v10 = &v20[13 * v22];
    do
    {
      if ( *v9 != -8192 && *v9 != -4096 )
      {
        v11 = v9[9];
        while ( v11 )
        {
          sub_253B2D0(*(_QWORD *)(v11 + 24));
          v12 = v11;
          v11 = *(_QWORD *)(v11 + 16);
          j_j___libc_free_0(v12);
        }
        v13 = v9[1];
        if ( (_QWORD *)v13 != v9 + 3 )
          _libc_free(v13);
      }
      v9 += 13;
    }
    while ( v10 != v9 );
    v8 = v22;
  }
  sub_C7D6A0((__int64)v20, 104 * v8, 8);
  return v7;
}
