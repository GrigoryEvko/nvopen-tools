// Function: sub_36D8790
// Address: 0x36d8790
//
_QWORD *__fastcall sub_36D8790(__int64 a1, __int64 a2, __m128i a3, __int64 a4, int a5)
{
  __int64 v5; // rbx
  __int64 v6; // rbp
  __int64 v7; // r12
  __int64 v11; // rax
  __int64 v12; // rsi
  __int64 v13; // r12
  __int64 v14; // r15
  int v15; // edx
  unsigned __int8 *v16; // rax
  __int64 v17; // rax
  __int64 v18; // rsi
  __int64 v19; // r12
  __int64 v20; // r15
  int v21; // edx
  __int64 v22; // r15
  __int64 v23; // rsi
  __int64 v24; // rdi
  __int64 v25; // rsi
  __int64 v26; // r12
  __int64 v27; // r13
  __int64 *v28; // rsi
  void *v29; // rbx
  int v30; // eax
  int v31; // eax
  unsigned __int64 v32; // [rsp-78h] [rbp-78h]
  unsigned int v33; // [rsp-70h] [rbp-70h]
  unsigned __int64 v34; // [rsp-68h] [rbp-68h] BYREF
  unsigned int v35; // [rsp-60h] [rbp-60h]
  void *v36; // [rsp-58h] [rbp-58h] BYREF
  int v37; // [rsp-50h] [rbp-50h]
  __int64 v38; // [rsp-30h] [rbp-30h]
  __int64 v39; // [rsp-28h] [rbp-28h]
  __int64 v40; // [rsp-8h] [rbp-8h]

  v40 = v6;
  v39 = v7;
  v38 = v5;
  switch ( a5 )
  {
    case 0:
      v23 = *(_QWORD *)(a2 + 80);
      v36 = (void *)v23;
      if ( v23 )
        sub_B96E90((__int64)&v36, v23, 1);
      v24 = *(_QWORD *)(a1 + 64);
      v37 = *(_DWORD *)(a2 + 72);
      v22 = (__int64)sub_3400BD0(v24, *(unsigned int *)(*(_QWORD *)(a1 + 1136) + 336LL), (__int64)&v36, 7, 0, 1u, a3, 0);
      if ( v36 )
        sub_B91220((__int64)&v36, (__int64)v36);
      return (_QWORD *)v22;
    case 1:
      v17 = *(_QWORD *)(a2 + 96);
      v18 = *(_QWORD *)(a2 + 80);
      v33 = 32;
      v32 = 1;
      v19 = *(_QWORD *)(a1 + 64);
      v36 = (void *)v18;
      v20 = v17 + 24;
      if ( v18 )
      {
        sub_B96E90((__int64)&v36, v18, 1);
        v21 = *(_DWORD *)(a2 + 72);
        v35 = v33;
        v37 = v21;
      }
      else
      {
        v31 = *(_DWORD *)(a2 + 72);
        v35 = 32;
        v37 = v31;
      }
      v34 = v32;
      sub_C47AC0((__int64)&v34, v20);
      v16 = sub_34007B0(v19, (__int64)&v34, (__int64)&v36, 7u, 0, 1u, a3, 0);
      goto LABEL_7;
    case 2:
      v11 = *(_QWORD *)(a2 + 96);
      v12 = *(_QWORD *)(a2 + 80);
      v33 = 16;
      v32 = 1;
      v13 = *(_QWORD *)(a1 + 64);
      v36 = (void *)v12;
      v14 = v11 + 24;
      if ( v12 )
      {
        sub_B96E90((__int64)&v36, v12, 1);
        v15 = *(_DWORD *)(a2 + 72);
        v35 = v33;
        v37 = v15;
      }
      else
      {
        v30 = *(_DWORD *)(a2 + 72);
        v35 = 16;
        v37 = v30;
      }
      v34 = v32;
      sub_C47AC0((__int64)&v34, v14);
      v16 = sub_34007B0(v13, (__int64)&v34, (__int64)&v36, 6u, 0, 1u, a3, 0);
LABEL_7:
      v22 = (__int64)v16;
      if ( v35 > 0x40 && v34 )
        j_j___libc_free_0_0(v34);
      if ( v36 )
        sub_B91220((__int64)&v36, (__int64)v36);
      if ( v33 > 0x40 && v32 )
        j_j___libc_free_0_0(v32);
      return (_QWORD *)v22;
    case 3:
      return sub_33EDBD0(
               *(_QWORD **)(a1 + 64),
               *(_DWORD *)(a2 + 96),
               **(unsigned __int16 **)(a2 + 48),
               *(_QWORD *)(*(_QWORD *)(a2 + 48) + 8LL),
               1);
    case 4:
      v25 = *(_QWORD *)(a2 + 80);
      v26 = *(_QWORD *)(a1 + 64);
      v34 = v25;
      if ( v25 )
        sub_B96E90((__int64)&v34, v25, 1);
      v27 = *(_QWORD *)(a2 + 96);
      v35 = *(_DWORD *)(a2 + 72);
      v28 = (__int64 *)(v27 + 24);
      v29 = sub_C33340();
      if ( *(void **)(v27 + 24) == v29 )
        sub_C3C790(&v36, (_QWORD **)v28);
      else
        sub_C33EB0(&v36, v28);
      if ( v36 == v29 )
        sub_C3CCB0((__int64)&v36);
      else
        sub_C34440((unsigned __int8 *)&v36);
      v22 = sub_33FE6E0(v26, (__int64 *)&v36, (__int64)&v34, 0xDu, 0, 1, a3);
      sub_91D830(&v36);
      if ( v34 )
        sub_B91220((__int64)&v34, v34);
      return (_QWORD *)v22;
    default:
      BUG();
  }
}
