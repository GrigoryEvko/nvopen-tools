// Function: sub_33E48F0
// Address: 0x33e48f0
//
__int64 __fastcall sub_33E48F0(
        _QWORD *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8,
        __int64 a9,
        __int64 a10)
{
  __int64 *v10; // r10
  __int64 v12; // rdi
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // rax
  _QWORD *v18; // rax
  __int64 *v19; // r10
  __int64 v20; // r8
  __int64 v21; // r9
  _QWORD *v22; // rbx
  __int64 v23; // r13
  __int64 v25; // rdx
  unsigned __int64 *v26; // rbx
  unsigned __int64 v27; // rcx
  _QWORD *v28; // r14
  _QWORD *v29; // r12
  __int64 v30; // rdx
  __int64 v31; // r13
  __int64 v32; // rax
  __int64 *v33; // r10
  int v34; // eax
  __int64 v35; // rax
  __int64 *v38; // [rsp+38h] [rbp-D8h]
  __int64 *v39; // [rsp+38h] [rbp-D8h]
  __int64 *v40; // [rsp+48h] [rbp-C8h] BYREF
  _QWORD *v41; // [rsp+50h] [rbp-C0h] BYREF
  int v42; // [rsp+58h] [rbp-B8h]
  _QWORD v43[22]; // [rsp+5Ch] [rbp-B4h] BYREF

  v10 = a1 + 22;
  v12 = (__int64)(a1 + 22);
  v43[0] = 0x400000020LL;
  v14 = (unsigned __int16)a2;
  if ( !(_WORD)a2 )
    v14 = a3;
  v43[1] = v14;
  v15 = (unsigned __int16)a4;
  if ( !(_WORD)a4 )
    v15 = a5;
  v43[2] = v15;
  v16 = (unsigned __int16)a7;
  if ( !(_WORD)a7 )
    v16 = a8;
  v43[3] = v16;
  v17 = (unsigned __int16)a9;
  if ( !(_WORD)a9 )
    v17 = a10;
  v38 = v10;
  v41 = (_QWORD *)((char *)v43 + 4);
  v43[4] = v17;
  v42 = 9;
  v40 = 0;
  v18 = sub_C65B40(v12, (__int64)&v41, (__int64 *)&v40, (__int64)off_4A367B0);
  v19 = v38;
  v20 = a9;
  v21 = a7;
  v22 = v18;
  if ( !v18 )
  {
    v25 = a1[24];
    a1[34] += 64LL;
    v26 = a1 + 24;
    v27 = ((v25 + 7) & 0xFFFFFFFFFFFFFFF8LL) + 64;
    if ( a1[25] >= v27 && v25 )
    {
      a1[24] = v27;
      v28 = (_QWORD *)((v25 + 7) & 0xFFFFFFFFFFFFFFF8LL);
    }
    else
    {
      v35 = sub_9D1E70((__int64)(a1 + 24), 64, 64, 3);
      v20 = a9;
      v21 = a7;
      v19 = v38;
      v28 = (_QWORD *)v35;
    }
    *v28 = a2;
    v28[2] = a4;
    v28[1] = a3;
    v28[4] = v21;
    v28[3] = a5;
    v28[6] = v20;
    v28[5] = a8;
    v39 = v19;
    v28[7] = a10;
    v29 = sub_C65D30((__int64)&v41, v26);
    v31 = v30;
    v32 = sub_A777F0(0x28u, (__int64 *)v26);
    v33 = v39;
    v22 = (_QWORD *)v32;
    if ( v32 )
    {
      *(_QWORD *)v32 = 0;
      *(_QWORD *)(v32 + 8) = v29;
      *(_QWORD *)(v32 + 16) = v31;
      *(_QWORD *)(v32 + 24) = v28;
      *(_DWORD *)(v32 + 32) = 4;
      v34 = sub_939680(v29, (__int64)v29 + 4 * v31);
      v33 = v39;
      *((_DWORD *)v22 + 9) = v34;
    }
    sub_C657C0(v33, v22, v40, (__int64)off_4A367B0);
  }
  v23 = v22[3];
  if ( v41 != (_QWORD *)((char *)v43 + 4) )
    _libc_free((unsigned __int64)v41);
  return v23;
}
