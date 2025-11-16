// Function: sub_319AD30
// Address: 0x319ad30
//
__int64 __fastcall sub_319AD30(__int64 a1, __int64 a2, __int64 a3, __int64 *a4)
{
  char v4; // r15
  __int64 v6; // rax
  unsigned __int64 v7; // rdx
  unsigned __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 *v11; // rax
  char v12; // r8
  __int64 v13; // r9
  _QWORD *v14; // r15
  __int64 v15; // r12
  unsigned int *v16; // rbx
  __int64 v17; // rdx
  unsigned int v18; // esi
  __int64 *v19; // rax
  char v20; // cl
  _QWORD *v21; // rax
  __int64 v22; // r9
  __int64 v23; // r12
  __int64 v24; // r13
  unsigned int *v25; // rbx
  unsigned int *v26; // r13
  __int64 v27; // rdx
  unsigned int v28; // esi
  __int64 v30; // [rsp-10h] [rbp-E0h]
  __int64 v31; // [rsp+8h] [rbp-C8h]
  char v32; // [rsp+14h] [rbp-BCh]
  __int64 v33; // [rsp+18h] [rbp-B8h]
  __int64 v34; // [rsp+18h] [rbp-B8h]
  unsigned int v35; // [rsp+20h] [rbp-B0h]
  char v36; // [rsp+28h] [rbp-A8h]
  char v37; // [rsp+2Fh] [rbp-A1h]
  char v38; // [rsp+30h] [rbp-A0h]
  _BYTE *v40[4]; // [rsp+40h] [rbp-90h] BYREF
  __int16 v41; // [rsp+60h] [rbp-70h]
  _QWORD v42[4]; // [rsp+70h] [rbp-60h] BYREF
  __int16 v43; // [rsp+90h] [rbp-40h]

  v4 = -1;
  v6 = *a4;
  v7 = (v6 | (1LL << **(_BYTE **)a1)) & -(v6 | (1LL << **(_BYTE **)a1));
  if ( v7 )
  {
    _BitScanReverse64(&v7, v7);
    v4 = 63 - (v7 ^ 0x3F);
  }
  v37 = -1;
  v8 = (v6 | (1LL << **(_BYTE **)(a1 + 8))) & -(v6 | (1LL << **(_BYTE **)(a1 + 8)));
  if ( v8 )
  {
    _BitScanReverse64(&v8, v8);
    v37 = 63 - (v8 ^ 0x3F);
  }
  v9 = sub_9208B0(*(_QWORD *)(a1 + 16), a2);
  v42[1] = v10;
  v42[0] = (unsigned __int64)(v9 + 7) >> 3;
  v35 = sub_CA1930(v42);
  v11 = *(__int64 **)(a1 + 40);
  v43 = 257;
  v40[0] = (_BYTE *)sub_AD64C0(*v11, *a4, 0);
  v31 = sub_921130((unsigned int **)a3, **(_QWORD **)(a1 + 24), **(_QWORD **)(a1 + 32), v40, 1, (__int64)v42, 3u);
  v41 = 257;
  v36 = v4;
  v12 = **(_BYTE **)(a1 + 48);
  v43 = 257;
  v32 = v12;
  v14 = sub_BD2C40(80, 1u);
  if ( v14 )
  {
    sub_B4D190((__int64)v14, a2, v31, (__int64)v42, v32, v36, 0, 0);
    v13 = v30;
  }
  (*(void (__fastcall **)(_QWORD, _QWORD *, _BYTE **, _QWORD, _QWORD, __int64))(**(_QWORD **)(a3 + 88) + 16LL))(
    *(_QWORD *)(a3 + 88),
    v14,
    v40,
    *(_QWORD *)(a3 + 56),
    *(_QWORD *)(a3 + 64),
    v13);
  if ( *(_QWORD *)a3 != *(_QWORD *)a3 + 16LL * *(unsigned int *)(a3 + 8) )
  {
    v33 = a3;
    v15 = *(_QWORD *)a3 + 16LL * *(unsigned int *)(a3 + 8);
    v16 = *(unsigned int **)a3;
    do
    {
      v17 = *((_QWORD *)v16 + 1);
      v18 = *v16;
      v16 += 4;
      sub_B99FD0((__int64)v14, v18, v17);
    }
    while ( (unsigned int *)v15 != v16 );
    a3 = v33;
  }
  v19 = *(__int64 **)(a1 + 40);
  v43 = 257;
  v40[0] = (_BYTE *)sub_AD64C0(*v19, *a4, 0);
  v34 = sub_921130((unsigned int **)a3, **(_QWORD **)(a1 + 24), **(_QWORD **)(a1 + 56), v40, 1, (__int64)v42, 3u);
  v20 = **(_BYTE **)(a1 + 64);
  v43 = 257;
  v38 = v20;
  v21 = sub_BD2C40(80, unk_3F10A10);
  v23 = (__int64)v21;
  if ( v21 )
    sub_B4D3C0((__int64)v21, (__int64)v14, v34, v38, v37, v22, 0, 0);
  (*(void (__fastcall **)(_QWORD, __int64, _QWORD *, _QWORD, _QWORD))(**(_QWORD **)(a3 + 88) + 16LL))(
    *(_QWORD *)(a3 + 88),
    v23,
    v42,
    *(_QWORD *)(a3 + 56),
    *(_QWORD *)(a3 + 64));
  v24 = 4LL * *(unsigned int *)(a3 + 8);
  v25 = *(unsigned int **)a3;
  v26 = &v25[v24];
  while ( v26 != v25 )
  {
    v27 = *((_QWORD *)v25 + 1);
    v28 = *v25;
    v25 += 4;
    sub_B99FD0(v23, v28, v27);
  }
  *a4 += v35;
  return v35;
}
