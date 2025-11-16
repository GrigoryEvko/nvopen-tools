// Function: sub_160FF80
// Address: 0x160ff80
//
void __fastcall sub_160FF80(__int64 a1, __int64 a2, __int64 a3, unsigned int a4)
{
  _QWORD *v5; // rbx
  unsigned int v6; // r14d
  __int64 v7; // r9
  _BYTE *v8; // rax
  __int64 v9; // rdx
  __int64 v10; // r13
  __int64 v11; // r12
  __int64 v12; // r12
  __int64 v13; // rax
  char *v14; // rbx
  char *v15; // r12
  char *v16; // rdi
  __int64 v17; // [rsp+8h] [rbp-3A8h]
  _QWORD v18[2]; // [rsp+20h] [rbp-390h] BYREF
  __int64 v19; // [rsp+30h] [rbp-380h] BYREF
  __int64 *v20; // [rsp+40h] [rbp-370h]
  __int64 v21; // [rsp+50h] [rbp-360h] BYREF
  _QWORD v22[2]; // [rsp+80h] [rbp-330h] BYREF
  __int64 v23; // [rsp+90h] [rbp-320h] BYREF
  __int64 *v24; // [rsp+A0h] [rbp-310h]
  __int64 v25; // [rsp+B0h] [rbp-300h] BYREF
  _QWORD v26[2]; // [rsp+E0h] [rbp-2D0h] BYREF
  __int64 v27; // [rsp+F0h] [rbp-2C0h] BYREF
  __int64 *v28; // [rsp+100h] [rbp-2B0h]
  __int64 v29; // [rsp+110h] [rbp-2A0h] BYREF
  __m128i v30; // [rsp+140h] [rbp-270h] BYREF
  __int64 v31; // [rsp+150h] [rbp-260h] BYREF
  __int64 *v32; // [rsp+160h] [rbp-250h]
  __int64 v33; // [rsp+170h] [rbp-240h] BYREF
  _QWORD v34[11]; // [rsp+1A0h] [rbp-210h] BYREF
  char *v35; // [rsp+1F8h] [rbp-1B8h]
  unsigned int v36; // [rsp+200h] [rbp-1B0h]
  char v37; // [rsp+208h] [rbp-1A8h] BYREF

  v5 = *(_QWORD **)(a3 + 32);
  if ( v5 != (_QWORD *)(a3 + 24) )
  {
    while ( 1 )
    {
      if ( !v5 )
        BUG();
      if ( v5 + 2 != (_QWORD *)(v5[2] & 0xFFFFFFFFFFFFFFF8LL) )
        break;
      v5 = (_QWORD *)v5[1];
      if ( v5 == (_QWORD *)(a3 + 24) )
        return;
    }
    v6 = sub_1633B40(a3);
    if ( v6 != a4 && !(*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a2 + 120LL))(a2) )
    {
      v7 = v5[3];
      if ( v7 )
        v7 -= 24;
      v17 = v6 - (unsigned __int64)a4;
      v30 = 0u;
      v31 = 0;
      sub_15CA680((__int64)v34, (__int64)"size-info", (__int64)"IRSizeChange", 12, &v30, v7);
      v8 = (_BYTE *)(*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a2 + 16LL))(a2);
      sub_15C9800((__int64)v18, "Pass", 4, v8, v9);
      v10 = sub_160FEC0((__int64)v34, (__int64)v18);
      sub_15CAB20(v10, ": IR instruction count changed from ", 0x24u);
      sub_15C9C50((__int64)v22, "IRInstrsBefore", 14, a4);
      v11 = sub_160FEC0(v10, (__int64)v22);
      sub_15CAB20(v11, " to ", 4u);
      sub_15C9C50((__int64)v26, "IRInstrsAfter", 13, v6);
      v12 = sub_160FEC0(v11, (__int64)v26);
      sub_15CAB20(v12, "; Delta: ", 9u);
      sub_15C9B00((__int64)&v30, "DeltaInstrCount", 15, v17);
      sub_160FEC0(v12, (__int64)&v30);
      if ( v32 != &v33 )
        j_j___libc_free_0(v32, v33 + 1);
      if ( (__int64 *)v30.m128i_i64[0] != &v31 )
        j_j___libc_free_0(v30.m128i_i64[0], v31 + 1);
      if ( v28 != &v29 )
        j_j___libc_free_0(v28, v29 + 1);
      if ( (__int64 *)v26[0] != &v27 )
        j_j___libc_free_0(v26[0], v27 + 1);
      if ( v24 != &v25 )
        j_j___libc_free_0(v24, v25 + 1);
      if ( (__int64 *)v22[0] != &v23 )
        j_j___libc_free_0(v22[0], v23 + 1);
      if ( v20 != &v21 )
        j_j___libc_free_0(v20, v21 + 1);
      if ( (__int64 *)v18[0] != &v19 )
        j_j___libc_free_0(v18[0], v19 + 1);
      v13 = sub_15E0530((__int64)(v5 - 7));
      sub_16027F0(v13, (__int64)v34);
      v14 = v35;
      v34[0] = &unk_49ECF68;
      v15 = &v35[88 * v36];
      if ( v35 != v15 )
      {
        do
        {
          v15 -= 88;
          v16 = (char *)*((_QWORD *)v15 + 4);
          if ( v16 != v15 + 48 )
            j_j___libc_free_0(v16, *((_QWORD *)v15 + 6) + 1LL);
          if ( *(char **)v15 != v15 + 16 )
            j_j___libc_free_0(*(_QWORD *)v15, *((_QWORD *)v15 + 2) + 1LL);
        }
        while ( v14 != v15 );
        v15 = v35;
      }
      if ( v15 != &v37 )
        _libc_free((unsigned __int64)v15);
    }
  }
}
