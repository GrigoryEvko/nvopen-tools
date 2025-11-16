// Function: sub_B9B4E0
// Address: 0xb9b4e0
//
__int64 *__fastcall sub_B9B4E0(__int64 *a1, __int64 a2)
{
  unsigned __int8 v4; // al
  __int64 v5; // rdx
  unsigned __int8 v6; // al
  __int64 v7; // rdx
  _BYTE *v8; // r13
  __int64 v9; // r14
  int v10; // ebx
  __int64 v11; // rax
  size_t v12; // rdx
  unsigned __int16 v13; // ax
  const void *v14; // rsi
  int v15; // r10d
  __int64 *v16; // rdi
  __int64 v17; // r11
  __int64 v18; // rax
  __int128 v20; // [rsp-28h] [rbp-D8h]
  const void *v21; // [rsp+0h] [rbp-B0h]
  int v22; // [rsp+0h] [rbp-B0h]
  size_t v23; // [rsp+8h] [rbp-A8h]
  __int64 v24; // [rsp+10h] [rbp-A0h]
  __int64 v25; // [rsp+18h] [rbp-98h]
  __int64 v26; // [rsp+20h] [rbp-90h]
  __int64 v27; // [rsp+28h] [rbp-88h]
  __int64 v28; // [rsp+30h] [rbp-80h]
  int v29; // [rsp+38h] [rbp-78h]
  int v30; // [rsp+3Ch] [rbp-74h]
  __int64 v31; // [rsp+48h] [rbp-68h]
  __int64 v32; // [rsp+50h] [rbp-60h]

  v4 = *(_BYTE *)(a2 - 16);
  if ( (v4 & 2) != 0 )
    v5 = *(_QWORD *)(a2 - 32);
  else
    v5 = a2 - 16 - 8LL * ((v4 >> 2) & 0xF);
  v24 = *(_QWORD *)(v5 + 40);
  v26 = *(_QWORD *)(v5 + 32);
  v29 = *(_DWORD *)(a2 + 20);
  v31 = sub_AF2E40(a2);
  v32 = *(_QWORD *)(a2 + 44);
  v27 = *(_QWORD *)(a2 + 32);
  v30 = sub_AF18D0(a2);
  v28 = *(_QWORD *)(a2 + 24);
  v6 = *(_BYTE *)(a2 - 16);
  if ( (v6 & 2) != 0 )
    v7 = *(_QWORD *)(a2 - 32);
  else
    v7 = a2 - 16 - 8LL * ((v6 >> 2) & 0xF);
  v25 = *(_QWORD *)(v7 + 24);
  v8 = *(_BYTE **)(v7 + 8);
  v9 = a2;
  v10 = *(_DWORD *)(a2 + 16);
  if ( *(_BYTE *)a2 != 16 )
    v9 = *(_QWORD *)sub_A17150((_BYTE *)(a2 - 16));
  v11 = sub_A547D0(a2, 2);
  v23 = v12;
  v21 = (const void *)v11;
  v13 = sub_AF18C0(a2);
  v14 = v21;
  v15 = v13;
  v16 = (__int64 *)(*(_QWORD *)(a2 + 8) & 0xFFFFFFFFFFFFFFF8LL);
  if ( (*(_QWORD *)(a2 + 8) & 4) != 0 )
    v16 = (__int64 *)*v16;
  v17 = 0;
  if ( v23 )
  {
    v22 = v13;
    v18 = sub_B9B140(v16, v14, v23);
    v15 = v22;
    v17 = v18;
  }
  *((_QWORD *)&v20 + 1) = v24;
  *(_QWORD *)&v20 = v26;
  *a1 = sub_B05AE0(v16, v15, v17, v9, v10, v8, v25, v28, v30, v27, v32, v31, v29, v20, 2u, 1);
  return a1;
}
