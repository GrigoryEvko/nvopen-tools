// Function: sub_20753D0
// Address: 0x20753d0
//
void __fastcall sub_20753D0(__int64 a1, __int64 a2, __m128i a3, __m128i a4, __m128i a5)
{
  int v6; // edx
  __int64 v7; // rax
  __int64 v8; // rsi
  unsigned __int8 v9; // r13
  unsigned int v10; // eax
  __int64 v11; // rdx
  __int64 *v12; // rax
  unsigned int v13; // edx
  unsigned __int8 v14; // r12
  __int64 v15; // r9
  __int64 v16; // rax
  _QWORD *v17; // r11
  __int64 *v18; // rax
  __int64 v19; // rdx
  __int64 *v20; // rsi
  __int64 v21; // rdx
  __int64 v22; // rdx
  __int128 v23; // rax
  __int64 *v24; // rax
  int v25; // edx
  int v26; // r13d
  __int64 *v27; // r12
  __int64 *v28; // rax
  __int64 v29; // r13
  __int128 v30; // [rsp+0h] [rbp-E0h]
  __int64 v31; // [rsp+10h] [rbp-D0h]
  __int64 v32; // [rsp+18h] [rbp-C8h]
  __int128 v33; // [rsp+20h] [rbp-C0h]
  __int128 v34; // [rsp+30h] [rbp-B0h]
  _QWORD *v35; // [rsp+40h] [rbp-A0h]
  int v36; // [rsp+48h] [rbp-98h]
  int v37; // [rsp+4Ch] [rbp-94h]
  __int64 v38; // [rsp+80h] [rbp-60h] BYREF
  int v39; // [rsp+88h] [rbp-58h]
  __int128 v40; // [rsp+90h] [rbp-50h] BYREF
  __int64 v41; // [rsp+A0h] [rbp-40h]

  v6 = *(_DWORD *)(a1 + 536);
  v7 = *(_QWORD *)a1;
  v38 = 0;
  v39 = v6;
  if ( v7 )
  {
    if ( &v38 != (__int64 *)(v7 + 48) )
    {
      v8 = *(_QWORD *)(v7 + 48);
      v38 = v8;
      if ( v8 )
        sub_1623A60((__int64)&v38, v8, 2);
    }
  }
  v9 = *(_BYTE *)(a2 + 56);
  v10 = *(unsigned __int16 *)(a2 + 18);
  v37 = (unsigned __int8)v10 >> 5;
  v36 = (v10 >> 2) & 7;
  *(_QWORD *)&v33 = sub_2051C20((__int64 *)a1, *(double *)a3.m128i_i64, *(double *)a4.m128i_i64, a5);
  *((_QWORD *)&v33 + 1) = v11;
  v12 = sub_20685E0(a1, *(__int64 **)(a2 - 48), a3, a4, a5);
  v14 = *(_BYTE *)(v12[5] + 16LL * v13);
  v16 = sub_1D25E70(*(_QWORD *)(a1 + 552), v14, 0, 2, 0, v15, 1, 0);
  v17 = *(_QWORD **)(a1 + 552);
  v31 = v16;
  v18 = *(__int64 **)(a2 - 72);
  v32 = v19;
  if ( v18 )
  {
    v40 = *(unsigned __int64 *)(a2 - 72);
    LOBYTE(v41) = 0;
    v18 = (__int64 *)*v18;
    if ( *((_BYTE *)v18 + 8) == 16 )
      v18 = *(__int64 **)v18[2];
    LODWORD(v18) = *((_DWORD *)v18 + 2) >> 8;
  }
  else
  {
    v40 = 0u;
    v41 = 0;
  }
  v20 = *(__int64 **)(a2 - 24);
  v35 = v17;
  HIDWORD(v41) = (_DWORD)v18;
  *(_QWORD *)&v34 = sub_20685E0(a1, v20, a3, a4, a5);
  *((_QWORD *)&v34 + 1) = v21;
  *(_QWORD *)&v30 = sub_20685E0(a1, *(__int64 **)(a2 - 48), a3, a4, a5);
  *((_QWORD *)&v30 + 1) = v22;
  *(_QWORD *)&v23 = sub_20685E0(a1, *(__int64 **)(a2 - 72), a3, a4, a5);
  v24 = sub_1D246E0(v35, 222, (__int64)&v38, v14, 0, 0, v31, v32, v33, v23, v30, v34, v40, v41, v36, v37, v9);
  v26 = v25;
  v27 = v24;
  *(_QWORD *)&v40 = a2;
  v28 = sub_205F5C0(a1 + 8, (__int64 *)&v40);
  v28[1] = (__int64)v27;
  *((_DWORD *)v28 + 4) = v26;
  v29 = *(_QWORD *)(a1 + 552);
  if ( v27 )
  {
    nullsub_686();
    *(_QWORD *)(v29 + 176) = v27;
    *(_DWORD *)(v29 + 184) = 2;
    sub_1D23870();
  }
  else
  {
    *(_QWORD *)(v29 + 176) = 0;
    *(_DWORD *)(v29 + 184) = 2;
  }
  if ( v38 )
    sub_161E7C0((__int64)&v38, v38);
}
