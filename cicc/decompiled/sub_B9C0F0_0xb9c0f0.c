// Function: sub_B9C0F0
// Address: 0xb9c0f0
//
__int64 *__fastcall sub_B9C0F0(__int64 *a1, __int64 a2)
{
  unsigned __int8 v3; // al
  _QWORD *v4; // rax
  int v5; // eax
  __int64 v6; // r15
  unsigned __int8 v7; // al
  __int64 v8; // rdx
  __int64 v9; // r13
  int v10; // r14d
  const void *v11; // rax
  size_t v12; // rdx
  __int64 *v13; // rdi
  __int64 v14; // rsi
  __int64 v16; // [rsp+8h] [rbp-68h]
  __int64 v17; // [rsp+10h] [rbp-60h]
  unsigned __int64 v18; // [rsp+18h] [rbp-58h]
  __int64 v19; // [rsp+20h] [rbp-50h]
  __int64 v20; // [rsp+28h] [rbp-48h]
  __int64 v21; // [rsp+30h] [rbp-40h]
  unsigned int v22; // [rsp+38h] [rbp-38h]
  int v23; // [rsp+3Ch] [rbp-34h]

  v3 = *(_BYTE *)(a2 - 16);
  if ( (v3 & 2) != 0 )
    v4 = *(_QWORD **)(a2 - 32);
  else
    v4 = (_QWORD *)(a2 - 16 - 8LL * ((v3 >> 2) & 0xF));
  v16 = v4[7];
  v17 = v4[6];
  v18 = v4[5];
  v19 = v4[4];
  v20 = v4[3];
  v22 = *(_DWORD *)(a2 + 20);
  v5 = sub_AF18D0(a2);
  v6 = *(_QWORD *)(a2 + 24);
  v23 = v5;
  v7 = *(_BYTE *)(a2 - 16);
  if ( (v7 & 2) != 0 )
    v8 = *(_QWORD *)(a2 - 32);
  else
    v8 = a2 - 16 - 8LL * ((v7 >> 2) & 0xF);
  v9 = a2;
  v10 = *(_DWORD *)(a2 + 16);
  v21 = *(_QWORD *)(v8 + 8);
  if ( *(_BYTE *)a2 != 16 )
    v9 = *(_QWORD *)sub_A17150((_BYTE *)(a2 - 16));
  v11 = (const void *)sub_A547D0(a2, 2);
  v13 = (__int64 *)(*(_QWORD *)(a2 + 8) & 0xFFFFFFFFFFFFFFF8LL);
  if ( (*(_QWORD *)(a2 + 8) & 4) != 0 )
    v13 = (__int64 *)*v13;
  v14 = 0;
  if ( v12 )
    v14 = sub_B9B140(v13, v11, v12);
  *a1 = sub_B03CA0(v13, v14, v9, v10, v21, v6, v23, v22, v20, v19, v18, v17, v16, 2u, 1);
  return a1;
}
