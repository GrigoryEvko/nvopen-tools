// Function: sub_205FE20
// Address: 0x205fe20
//
void __fastcall sub_205FE20(__int64 a1, _QWORD *a2, double a3, double a4, __m128i a5)
{
  __int64 v7; // r15
  __int64 v8; // r13
  __int64 v9; // rdi
  __int64 v10; // rax
  __int64 *v11; // r14
  __int64 *v12; // r13
  int v13; // eax
  int v14; // edx
  __int64 v15; // rax
  __int64 v16; // rsi
  __int64 *v17; // rax
  int v18; // edx
  const void ***v19; // r14
  int v20; // edx
  __int64 v21; // r8
  _QWORD *v22; // rax
  __int64 v23; // rdx
  __int64 *v24; // r13
  int v25; // edx
  int v26; // r14d
  __int64 *v27; // rax
  _BYTE *v28; // rdi
  __int128 v29; // [rsp-10h] [rbp-160h]
  __int64 v30; // [rsp+0h] [rbp-150h]
  __int64 v31; // [rsp+8h] [rbp-148h]
  int v32; // [rsp+10h] [rbp-140h]
  int v33; // [rsp+28h] [rbp-128h]
  __int64 v34; // [rsp+30h] [rbp-120h]
  int v35; // [rsp+38h] [rbp-118h]
  __int64 v36; // [rsp+50h] [rbp-100h] BYREF
  int v37; // [rsp+58h] [rbp-F8h]
  __int64 v38[4]; // [rsp+60h] [rbp-F0h] BYREF
  _QWORD *v39; // [rsp+80h] [rbp-D0h] BYREF
  int v40; // [rsp+88h] [rbp-C8h]
  _QWORD *v41; // [rsp+90h] [rbp-C0h]
  __int64 v42; // [rsp+98h] [rbp-B8h]
  _QWORD v43[2]; // [rsp+A0h] [rbp-B0h] BYREF
  _BYTE v44[32]; // [rsp+B0h] [rbp-A0h] BYREF
  unsigned __int64 v45[2]; // [rsp+D0h] [rbp-80h] BYREF
  _BYTE v46[112]; // [rsp+E0h] [rbp-70h] BYREF

  v7 = *(a2 - 3);
  v8 = *a2;
  memset(v38, 0, 24);
  sub_14A8180((__int64)a2, v38, 0);
  v45[0] = (unsigned __int64)v46;
  v45[1] = 0x400000000LL;
  v43[1] = 0x400000000LL;
  v9 = *(_QWORD *)(*(_QWORD *)(a1 + 552) + 32LL);
  v43[0] = v44;
  v10 = sub_1E0A0C0(v9);
  sub_20C7CE0(*(_QWORD *)(*(_QWORD *)(a1 + 552) + 16LL), v10, v8, v45, v43, 0);
  v11 = (__int64 *)v45[0];
  v12 = *(__int64 **)(a1 + 552);
  v13 = sub_1FE6270(*(_QWORD *)(a1 + 712), (__int64)a2, *(_QWORD *)(*(_QWORD *)(a1 + 712) + 784LL), v7);
  v14 = *(_DWORD *)(a1 + 536);
  v36 = 0;
  v35 = v13;
  v15 = *(_QWORD *)a1;
  v37 = v14;
  if ( v15 )
  {
    if ( &v36 != (__int64 *)(v15 + 48) )
    {
      v16 = *(_QWORD *)(v15 + 48);
      v36 = v16;
      if ( v16 )
        sub_1623A60((__int64)&v36, v16, 2);
    }
  }
  v17 = sub_2051C20((__int64 *)a1, a3, a4, a5);
  v33 = v18;
  v30 = *v11;
  v31 = v11[1];
  v34 = (__int64)v17;
  v19 = (const void ***)sub_1D252B0((__int64)v12, (unsigned int)*v11, v31, 1, 0);
  v32 = v20;
  v39 = (_QWORD *)v34;
  v40 = v33;
  v22 = sub_1D2A660(v12, v35, v30, v31, v21, v31);
  v42 = v23;
  *((_QWORD *)&v29 + 1) = 2;
  *(_QWORD *)&v29 = &v39;
  v41 = v22;
  v24 = sub_1D36D80(v12, 47, (__int64)&v36, v19, v32, a3, a4, a5, (__int64)&v39, v29);
  v26 = v25;
  if ( v36 )
    sub_161E7C0((__int64)&v36, v36);
  v39 = a2;
  v27 = sub_205F5C0(a1 + 8, (__int64 *)&v39);
  v28 = (_BYTE *)v43[0];
  v27[1] = (__int64)v24;
  *((_DWORD *)v27 + 4) = v26;
  if ( v28 != v44 )
    _libc_free((unsigned __int64)v28);
  if ( (_BYTE *)v45[0] != v46 )
    _libc_free(v45[0]);
}
