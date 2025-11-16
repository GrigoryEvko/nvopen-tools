// Function: sub_2E20270
// Address: 0x2e20270
//
void __fastcall sub_2E20270(_QWORD *a1, __int64 *a2, __int64 a3, unsigned int a4, __int64 *a5, __int64 a6)
{
  __int64 v6; // rax
  unsigned __int64 v8; // rdx
  __int64 v13; // r11
  int v14; // eax
  unsigned __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rax
  __int64 v18; // r9
  char v19; // dl
  __int64 v20; // rsi
  _QWORD *v21; // rdi
  __int64 v22; // [rsp+0h] [rbp-50h]
  __int64 v24[7]; // [rsp+18h] [rbp-38h] BYREF

  v6 = a3 >> 1;
  v8 = a3 & 0xFFFFFFFFFFFFFFF8LL;
  v13 = a1[2];
  v14 = v6 & 3;
  if ( v14 )
  {
    v17 = v8 | (2LL * (v14 - 1));
    v16 = *(_QWORD *)((v17 & 0xFFFFFFFFFFFFFFF8LL) + 16);
    if ( v16 )
      goto LABEL_3;
LABEL_7:
    v20 = *(unsigned int *)(v13 + 304);
    v21 = *(_QWORD **)(v13 + 296);
    v24[0] = v17;
    v18 = *(sub_2E1D5D0(v21, (__int64)&v21[2 * v20], v24) - 1);
    goto LABEL_4;
  }
  v15 = *(_QWORD *)v8 & 0xFFFFFFFFFFFFFFF8LL;
  v16 = *(_QWORD *)(v15 + 16);
  v17 = v15 | 6;
  if ( !v16 )
    goto LABEL_7;
LABEL_3:
  v18 = *(_QWORD *)(v16 + 24);
LABEL_4:
  v22 = v18;
  if ( !sub_2E0E770((__int64)a2, a5, a6, *(_QWORD *)(*(_QWORD *)(v13 + 152) + 16LL * *(unsigned int *)(v18 + 24)), a3)
    && !v19
    && !(unsigned __int8)sub_2E1F4C0(a1, (__int64)a2, v22, a3, (__int64 *)a4, v22, a5, a6) )
  {
    sub_2E1E760(a1, a2);
  }
}
