// Function: sub_2060C70
// Address: 0x2060c70
//
void __fastcall sub_2060C70(__int64 a1, __int64 a2, double a3, double a4, __m128i a5)
{
  __int64 v6; // rdi
  __int64 v7; // rdx
  __int64 v8; // r13
  unsigned int v9; // eax
  unsigned int v10; // edx
  _QWORD *v11; // rbx
  _QWORD *v12; // r13
  __int64 v13; // rax
  __int64 v14; // rax
  unsigned int *v15; // rsi
  __int64 *v16; // r13
  __int64 v17; // rdx
  __int64 v18; // rcx
  __int64 v19; // r8
  int v20; // r9d
  __int64 v21; // rdx
  __int128 v22; // rcx
  __int64 v23; // rax
  __int64 v24; // rsi
  __int64 v25; // rbx
  int v26; // edx
  int v27; // r13d
  __int64 v28; // r12
  __int64 v29; // [rsp+0h] [rbp-90h]
  __int64 v30; // [rsp+30h] [rbp-60h] BYREF
  int v31; // [rsp+38h] [rbp-58h]
  _BYTE *v32; // [rsp+40h] [rbp-50h] BYREF
  __int64 v33; // [rsp+48h] [rbp-48h]
  _BYTE v34[64]; // [rsp+50h] [rbp-40h] BYREF

  v32 = v34;
  v6 = *(_QWORD *)(a1 + 712);
  v33 = 0x100000000LL;
  if ( (*(_BYTE *)(a2 + 18) & 1) != 0 )
  {
    v7 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
    v8 = *(_QWORD *)(a2 + 24 * (1 - v7));
    if ( *(_QWORD *)(v6 + 32) && v8 )
    {
      v9 = sub_13774B0(*(_QWORD *)(v6 + 32), *(_QWORD *)(*(_QWORD *)(v6 + 784) + 40LL), *(_QWORD *)(a2 + 24 * (1 - v7)));
      v6 = *(_QWORD *)(a1 + 712);
      v10 = v9;
      goto LABEL_7;
    }
  }
  else
  {
    v8 = 0;
  }
  v10 = 0;
LABEL_7:
  sub_2060560(v6, v8, v10, (__int64)&v32);
  v11 = v32;
  v12 = &v32[16 * (unsigned int)v33];
  if ( v12 != (_QWORD *)v32 )
  {
    do
    {
      v13 = *v11;
      v11 += 2;
      *(_BYTE *)(v13 + 180) = 1;
      sub_2052F00(a1, *(_QWORD *)(*(_QWORD *)(a1 + 712) + 784LL), *(v11 - 2), *((_DWORD *)v11 - 2));
    }
    while ( v12 != v11 );
  }
  v14 = *(_QWORD *)(*(_QWORD *)(a1 + 712) + 784LL);
  v15 = *(unsigned int **)(v14 + 120);
  sub_1D96570(*(unsigned int **)(v14 + 112), v15);
  v16 = *(__int64 **)(a1 + 552);
  v30 = 0;
  *(_QWORD *)&v22 = sub_2051DF0((__int64 *)a1, a3, a4, a5, (__int64)v15, v17, v18, v19, v20);
  *((_QWORD *)&v22 + 1) = v21;
  v23 = *(_QWORD *)a1;
  v31 = *(_DWORD *)(a1 + 536);
  if ( v23 )
  {
    if ( &v30 != (__int64 *)(v23 + 48) )
    {
      v24 = *(_QWORD *)(v23 + 48);
      v30 = v24;
      if ( v24 )
      {
        v29 = v22;
        sub_1623A60((__int64)&v30, v24, 2);
        *(_QWORD *)&v22 = v29;
      }
    }
  }
  v25 = sub_1D309E0(v16, 198, (__int64)&v30, 1, 0, 0, a3, a4, *(double *)a5.m128i_i64, v22);
  v27 = v26;
  if ( v30 )
    sub_161E7C0((__int64)&v30, v30);
  v28 = *(_QWORD *)(a1 + 552);
  if ( v25 )
  {
    nullsub_686();
    *(_QWORD *)(v28 + 176) = v25;
    *(_DWORD *)(v28 + 184) = v27;
    sub_1D23870();
  }
  else
  {
    *(_QWORD *)(v28 + 176) = 0;
    *(_DWORD *)(v28 + 184) = v27;
  }
  if ( v32 != v34 )
    _libc_free((unsigned __int64)v32);
}
