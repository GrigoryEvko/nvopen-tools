// Function: sub_31397B0
// Address: 0x31397b0
//
__int64 *__fastcall sub_31397B0(__int64 *a1, __int64 **a2, __int64 a3)
{
  __int64 *v3; // r12
  __int64 v4; // r14
  __int64 v5; // r15
  char v6; // cl
  char v7; // bl
  __int64 v8; // rax
  __int64 v10; // rax
  __int64 *v11; // r13
  __int64 v12; // rdx
  __int16 v13; // dx
  __int64 v14; // rdi
  __int16 v15; // ax
  __int64 v16; // r14
  __int64 v17; // r15
  __int64 v18; // rbx
  _QWORD *v19; // rax
  __int64 v20; // r13
  unsigned int *v21; // rbx
  unsigned int *v22; // r14
  __int64 v23; // rdx
  char v24; // [rsp+Fh] [rbp-B1h]
  __int64 v25; // [rsp+10h] [rbp-B0h]
  char v26; // [rsp+10h] [rbp-B0h]
  __int64 *v27; // [rsp+18h] [rbp-A8h]
  char v28[32]; // [rsp+20h] [rbp-A0h] BYREF
  __int16 v29; // [rsp+40h] [rbp-80h]
  __int64 v30; // [rsp+50h] [rbp-70h] BYREF
  __int64 v31; // [rsp+58h] [rbp-68h] BYREF
  __int64 v32; // [rsp+60h] [rbp-60h]
  __int64 v33; // [rsp+68h] [rbp-58h]
  __int64 v34; // [rsp+70h] [rbp-50h]
  __int16 v35; // [rsp+78h] [rbp-48h]
  __int64 v36[8]; // [rsp+80h] [rbp-40h] BYREF

  v3 = a1;
  v4 = *(_QWORD *)a3;
  v5 = *(_QWORD *)(a3 + 8);
  v6 = *(_BYTE *)(a3 + 16);
  v27 = *a2;
  v7 = *(_BYTE *)(a3 + 17);
  if ( v5 == *(_QWORD *)a3 + 48LL )
  {
    v10 = *v27;
    v11 = (__int64 *)(*v27 + 512);
    v30 = (__int64)v11;
    v12 = *(_QWORD *)(v10 + 560);
    v31 = 0;
    v32 = 0;
    v33 = v12;
    if ( v12 != 0 && v12 != -4096 && v12 != -8192 )
    {
      v24 = v6;
      v25 = v10;
      sub_BD73F0((__int64)&v31);
      v6 = v24;
      v10 = v25;
    }
    v13 = *(_WORD *)(v10 + 576);
    v26 = v6;
    v34 = *(_QWORD *)(v10 + 568);
    v35 = v13;
    sub_B33910(v36, v11);
    v14 = *v27;
    if ( v4 )
    {
      LOBYTE(v15) = v26;
      HIBYTE(v15) = v7;
      sub_A88F30(v14 + 512, v4, v5, v15);
    }
    else
    {
      *(_QWORD *)(v14 + 560) = 0;
      *(_QWORD *)(v14 + 568) = 0;
      *(_WORD *)(v14 + 576) = 0;
    }
    v16 = *v27;
    v17 = *(_QWORD *)v27[1];
    v18 = *v27 + 512;
    v29 = 257;
    v19 = sub_BD2C40(72, 1u);
    v20 = (__int64)v19;
    if ( v19 )
      sub_B4C8F0((__int64)v19, v17, 1u, 0, 0);
    a2 = (__int64 **)v20;
    (*(void (__fastcall **)(_QWORD, __int64, char *, _QWORD, _QWORD))(**(_QWORD **)(v16 + 600) + 16LL))(
      *(_QWORD *)(v16 + 600),
      v20,
      v28,
      *(_QWORD *)(v18 + 56),
      *(_QWORD *)(v18 + 64));
    v21 = *(unsigned int **)(v16 + 512);
    v22 = &v21[4 * *(unsigned int *)(v16 + 520)];
    while ( v22 != v21 )
    {
      v23 = *((_QWORD *)v21 + 1);
      a2 = (__int64 **)*v21;
      v21 += 4;
      sub_B99FD0(v20, (unsigned int)a2, v23);
    }
    a1 = &v30;
    v4 = *(_QWORD *)(v20 + 40);
    v5 = v20 + 24;
    v7 = 0;
    sub_F11320((__int64)&v30);
    v6 = 0;
  }
  v8 = v27[2];
  v30 = v4;
  v31 = v5;
  LOBYTE(v32) = v6;
  BYTE1(v32) = v7;
  if ( !*(_QWORD *)(v8 + 16) )
    sub_4263D6(a1, a2, a3);
  (*(void (__fastcall **)(__int64 *, __int64, __int64 *))(v8 + 24))(v3, v8, &v30);
  return v3;
}
