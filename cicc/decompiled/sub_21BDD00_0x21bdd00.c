// Function: sub_21BDD00
// Address: 0x21bdd00
//
_QWORD *__fastcall sub_21BDD00(__int64 a1, __int64 a2, __m128i a3, double a4, __m128i a5, __int64 a6, int a7)
{
  __int64 v8; // rax
  __int64 v9; // rsi
  __int64 v10; // r15
  __int64 v11; // r13
  int v12; // edx
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // rsi
  __int64 v16; // r15
  __int64 v17; // r13
  int v18; // edx
  _QWORD *v19; // r12
  __int64 v20; // rsi
  __int64 v21; // r12
  __int64 v22; // rdx
  _QWORD *v23; // rax
  __int64 v24; // rsi
  __int64 v25; // rsi
  __int64 v26; // rdx
  __int64 v27; // rax
  __int64 v28; // rsi
  __int64 v30; // rsi
  __int64 v31; // r12
  int v32; // eax
  __int64 v33; // rbx
  __int64 *v34; // rdi
  double v35; // xmm0_8
  _QWORD *v36; // rax
  int v37; // eax
  int v38; // eax
  __int64 v39; // [rsp+0h] [rbp-60h]
  unsigned int v40; // [rsp+8h] [rbp-58h]
  __int64 v41; // [rsp+10h] [rbp-50h] BYREF
  unsigned int v42; // [rsp+18h] [rbp-48h]
  __int64 v43; // [rsp+20h] [rbp-40h] BYREF
  int v44; // [rsp+28h] [rbp-38h]

  switch ( a7 )
  {
    case 0:
      v25 = *(_QWORD *)(a2 + 72);
      v21 = *(_QWORD *)(a1 + 272);
      v43 = v25;
      if ( v25 )
        sub_1623A60((__int64)&v43, v25, 2);
      v26 = *(_QWORD *)(a2 + 88);
      v44 = *(_DWORD *)(a2 + 64);
      v23 = *(_QWORD **)(v26 + 24);
      if ( *(_DWORD *)(v26 + 32) > 0x40u )
        v23 = (_QWORD *)*v23;
      v24 = 32;
      goto LABEL_24;
    case 1:
      v20 = *(_QWORD *)(a2 + 72);
      v21 = *(_QWORD *)(a1 + 272);
      v43 = v20;
      if ( v20 )
        sub_1623A60((__int64)&v43, v20, 2);
      v22 = *(_QWORD *)(a2 + 88);
      v44 = *(_DWORD *)(a2 + 64);
      v23 = *(_QWORD **)(v22 + 24);
      if ( *(_DWORD *)(v22 + 32) > 0x40u )
        v23 = (_QWORD *)*v23;
      v24 = 64;
LABEL_24:
      v27 = sub_1D38BB0(v21, v24 - (_QWORD)v23, (__int64)&v43, 5, 0, 1, a3, a4, a5, 0);
      v28 = v43;
      v19 = (_QWORD *)v27;
      if ( v43 )
        goto LABEL_25;
      return v19;
    case 2:
      v14 = *(_QWORD *)(a2 + 88);
      v15 = *(_QWORD *)(a2 + 72);
      v40 = 32;
      v39 = 1;
      v16 = *(_QWORD *)(a1 + 272);
      v43 = v15;
      v17 = v14 + 24;
      if ( v15 )
      {
        sub_1623A60((__int64)&v43, v15, 2);
        v18 = *(_DWORD *)(a2 + 64);
        v42 = v40;
        v44 = v18;
      }
      else
      {
        v38 = *(_DWORD *)(a2 + 64);
        v42 = 32;
        v44 = v38;
      }
      v41 = v39;
      sub_16A7E20((__int64)&v41, v17);
      v13 = sub_1D38970(v16, (__int64)&v41, (__int64)&v43, 5u, 0, 1u, a3, a4, a5, 0);
      goto LABEL_6;
    case 3:
      v8 = *(_QWORD *)(a2 + 88);
      v9 = *(_QWORD *)(a2 + 72);
      v40 = 16;
      v39 = 1;
      v10 = *(_QWORD *)(a1 + 272);
      v43 = v9;
      v11 = v8 + 24;
      if ( v9 )
      {
        sub_1623A60((__int64)&v43, v9, 2);
        v12 = *(_DWORD *)(a2 + 64);
        v42 = v40;
        v44 = v12;
      }
      else
      {
        v37 = *(_DWORD *)(a2 + 64);
        v42 = 16;
        v44 = v37;
      }
      v41 = v39;
      sub_16A7E20((__int64)&v41, v11);
      v13 = sub_1D38970(v10, (__int64)&v41, (__int64)&v43, 4u, 0, 1u, a3, a4, a5, 0);
LABEL_6:
      v19 = (_QWORD *)v13;
      if ( v42 > 0x40 && v41 )
        j_j___libc_free_0_0(v41);
      if ( v43 )
        sub_161E7C0((__int64)&v43, v43);
      if ( v40 > 0x40 && v39 )
        j_j___libc_free_0_0(v39);
      break;
    case 4:
      v30 = *(_QWORD *)(a2 + 72);
      v31 = *(_QWORD *)(a1 + 272);
      v43 = v30;
      if ( v30 )
        sub_1623A60((__int64)&v43, v30, 2);
      v32 = *(_DWORD *)(a2 + 64);
      v33 = *(_QWORD *)(a2 + 88);
      v44 = v32;
      v34 = (__int64 *)(v33 + 32);
      if ( *(void **)(v33 + 32) == sub_16982C0() )
        v34 = (__int64 *)(*(_QWORD *)(v33 + 40) + 8LL);
      v35 = sub_169D8E0(v34);
      v36 = sub_1D364E0(v31, (__int64)&v43, 10, 0, 1u, -v35, a4, a5);
      v28 = v43;
      v19 = v36;
      if ( v43 )
LABEL_25:
        sub_161E7C0((__int64)&v43, v28);
      break;
  }
  return v19;
}
