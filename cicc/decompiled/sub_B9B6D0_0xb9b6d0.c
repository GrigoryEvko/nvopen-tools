// Function: sub_B9B6D0
// Address: 0xb9b6d0
//
__int64 *__fastcall sub_B9B6D0(__int64 *a1, __int64 a2)
{
  __int64 v3; // rbx
  size_t v4; // rdx
  size_t v5; // r13
  size_t v6; // rdx
  size_t v7; // r12
  __int64 *v8; // rdi
  __int64 v9; // rcx
  __int64 v10; // rdx
  __int64 v11; // rax
  __int64 v13; // [rsp+0h] [rbp-F0h]
  const void *v14; // [rsp+8h] [rbp-E8h]
  const void *v15; // [rsp+10h] [rbp-E0h]
  __int64 v16; // [rsp+18h] [rbp-D8h]
  __int64 v17; // [rsp+20h] [rbp-D0h]
  unsigned __int64 v18; // [rsp+28h] [rbp-C8h]
  __int64 v19; // [rsp+30h] [rbp-C0h]
  __int64 v20; // [rsp+38h] [rbp-B8h]
  __int64 v21; // [rsp+40h] [rbp-B0h]
  __int64 v22; // [rsp+48h] [rbp-A8h]
  __int64 v23; // [rsp+50h] [rbp-A0h]
  __int64 v24; // [rsp+58h] [rbp-98h]
  __int64 v25; // [rsp+60h] [rbp-90h]
  __int64 v26; // [rsp+68h] [rbp-88h]
  __int64 v27; // [rsp+70h] [rbp-80h]
  int v28; // [rsp+78h] [rbp-78h]
  int v29; // [rsp+7Ch] [rbp-74h]
  __int64 v30; // [rsp+80h] [rbp-70h]
  __int64 v31; // [rsp+88h] [rbp-68h]
  int v32; // [rsp+90h] [rbp-60h]
  unsigned int v33; // [rsp+94h] [rbp-5Ch]
  int v34; // [rsp+98h] [rbp-58h]
  unsigned int v35; // [rsp+9Ch] [rbp-54h]
  __int64 v36; // [rsp+A0h] [rbp-50h]

  v3 = a2;
  v28 = *(_DWORD *)(a2 + 40);
  v16 = *((_QWORD *)sub_A17150((_BYTE *)(a2 - 16)) + 14);
  v17 = *((_QWORD *)sub_A17150((_BYTE *)(a2 - 16)) + 13);
  v24 = *((_QWORD *)sub_A17150((_BYTE *)(a2 - 16)) + 12);
  v25 = *((_QWORD *)sub_A17150((_BYTE *)(a2 - 16)) + 11);
  v26 = *((_QWORD *)sub_A17150((_BYTE *)(a2 - 16)) + 10);
  v27 = *((_QWORD *)sub_A17150((_BYTE *)(a2 - 16)) + 9);
  v18 = *((_QWORD *)sub_A17150((_BYTE *)(a2 - 16)) + 8);
  v15 = (const void *)sub_A547D0(a2, 7);
  v5 = v4;
  v19 = *((_QWORD *)sub_A17150((_BYTE *)(a2 - 16)) + 6);
  v20 = *((_QWORD *)sub_A17150((_BYTE *)(a2 - 16)) + 5);
  v36 = *(_QWORD *)(a2 + 48);
  v29 = *(_DWORD *)(a2 + 44);
  v21 = *((_QWORD *)sub_A17150((_BYTE *)(a2 - 16)) + 4);
  v33 = *(_DWORD *)(a2 + 20);
  v31 = *(_QWORD *)(a2 + 32);
  v35 = sub_AF18D0(a2);
  v30 = *(_QWORD *)(a2 + 24);
  v22 = *((_QWORD *)sub_A17150((_BYTE *)(a2 - 16)) + 3);
  v23 = *((_QWORD *)sub_A17150((_BYTE *)(a2 - 16)) + 1);
  v32 = *(_DWORD *)(a2 + 16);
  if ( *(_BYTE *)a2 != 16 )
    v3 = *(_QWORD *)sub_A17150((_BYTE *)(a2 - 16));
  v14 = (const void *)sub_A547D0(a2, 2);
  v7 = v6;
  v34 = (unsigned __int16)sub_AF18C0(a2);
  v8 = (__int64 *)(*(_QWORD *)(a2 + 8) & 0xFFFFFFFFFFFFFFF8LL);
  if ( (*(_QWORD *)(a2 + 8) & 4) != 0 )
    v8 = (__int64 *)*v8;
  v9 = 0;
  if ( v5 )
    v9 = sub_B9B140(v8, v15, v5);
  v10 = 0;
  if ( v7 )
  {
    v13 = v9;
    v11 = sub_B9B140(v8, v14, v7);
    v9 = v13;
    v10 = v11;
  }
  *a1 = sub_B065E0(
          v8,
          v34,
          v10,
          v3,
          v32,
          v23,
          v22,
          v30,
          v35,
          v31,
          v33,
          v21,
          v29,
          v36,
          v20,
          v19,
          v9,
          v18,
          v27,
          v26,
          v25,
          v24,
          v17,
          v16,
          v28,
          2u,
          1);
  return a1;
}
