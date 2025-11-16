// Function: sub_2D5CA10
// Address: 0x2d5ca10
//
void __fastcall sub_2D5CA10(__int64 a1, unsigned __int64 a2, char a3)
{
  __int64 ***v5; // rax
  __int64 *v6; // r15
  __int64 **v7; // r13
  int v8; // r14d
  unsigned int v9; // esi
  unsigned __int64 v10; // rax
  __int64 v11; // rdi
  __int64 v12; // r15
  char v13; // r12
  char v14; // dl
  __int64 *v15; // r12
  char v16; // bl
  _QWORD *v17; // rax
  __int64 v18; // r9
  __int64 v19; // r13
  __int64 v20; // rbx
  __int64 v21; // r12
  __int64 v22; // rdx
  unsigned int v23; // esi
  unsigned int **v24; // r13
  _QWORD *v25; // rax
  __int64 v26; // rax
  __int64 v27; // rax
  __int64 v28; // rcx
  unsigned __int64 v29; // rax
  __int64 v31; // [rsp+8h] [rbp-78h]
  _BYTE *v32; // [rsp+18h] [rbp-68h] BYREF
  _BYTE v33[32]; // [rsp+20h] [rbp-60h] BYREF
  __int16 v34; // [rsp+40h] [rbp-40h]

  v5 = *(__int64 ****)(a1 + 8);
  v6 = *(__int64 **)a1;
  v34 = 257;
  v7 = *v5;
  v8 = sub_BCB060(*(_QWORD *)(a2 + 8));
  v9 = 49;
  if ( v8 != (unsigned int)sub_BCB060((__int64)v7) )
    v9 = 39;
  v10 = sub_2D5B7B0(v6, v9, a2, v7, (__int64)v33, 0, (int)v32, 0);
  v11 = *(_QWORD *)(a1 + 16);
  v31 = v10;
  v12 = *(_QWORD *)(v11 - 32);
  _BitScanReverse64(&v10, 1LL << (*(_WORD *)(v11 + 2) >> 1));
  v13 = 63 - (v10 ^ 0x3F);
  v14 = v13;
  if ( **(_BYTE **)(a1 + 24) )
  {
    if ( !a3 )
      goto LABEL_5;
  }
  else if ( a3 )
  {
    goto LABEL_5;
  }
  v24 = *(unsigned int ***)a1;
  v34 = 257;
  v25 = (_QWORD *)sub_BD5C60(v11);
  v26 = sub_BCB2D0(v25);
  v32 = (_BYTE *)sub_ACD640(v26, 1, 0);
  v27 = sub_921130(v24, **(_QWORD **)(a1 + 8), v12, &v32, 1, (__int64)v33, 0);
  v14 = -1;
  v12 = v27;
  v28 = -(__int64)((**(_DWORD **)(a1 + 32) >> 3) | (unsigned __int64)(1LL << v13));
  if ( (v28 & ((**(_DWORD **)(a1 + 32) >> 3) | (unsigned __int64)(1LL << v13))) != 0 )
  {
    _BitScanReverse64(&v29, v28 & ((**(_DWORD **)(a1 + 32) >> 3) | (unsigned __int64)(1LL << v13)));
    v14 = 63 - (v29 ^ 0x3F);
  }
LABEL_5:
  v15 = *(__int64 **)a1;
  v34 = 257;
  v16 = v14;
  v17 = sub_BD2C40(80, unk_3F10A10);
  v19 = (__int64)v17;
  if ( v17 )
    sub_B4D3C0((__int64)v17, v31, v12, 0, v16, v18, 0, 0);
  (*(void (__fastcall **)(__int64, __int64, _BYTE *, __int64, __int64))(*(_QWORD *)v15[11] + 16LL))(
    v15[11],
    v19,
    v33,
    v15[7],
    v15[8]);
  v20 = *v15;
  v21 = *v15 + 16LL * *((unsigned int *)v15 + 2);
  while ( v21 != v20 )
  {
    v22 = *(_QWORD *)(v20 + 8);
    v23 = *(_DWORD *)v20;
    v20 += 16;
    sub_B99FD0(v19, v23, v22);
  }
}
