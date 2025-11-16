// Function: sub_9417D0
// Address: 0x9417d0
//
_QWORD *__fastcall sub_9417D0(__int64 a1, __int64 a2, __int64 a3)
{
  const char *v5; // r14
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // r8
  int v9; // r9d
  __int64 v10; // rax
  __int64 v11; // rax
  _BOOL8 v12; // rdi
  int v13; // eax
  int v14; // edx
  int v15; // ecx
  int v16; // r8d
  int v17; // r9d
  int v18; // eax
  __int64 v19; // rax
  int v20; // edx
  _QWORD *result; // rax
  int v22; // [rsp+0h] [rbp-90h]
  int v23; // [rsp+8h] [rbp-88h]
  __int64 v24; // [rsp+10h] [rbp-80h]
  __int64 v25; // [rsp+18h] [rbp-78h]
  int v26; // [rsp+20h] [rbp-70h]
  int v27; // [rsp+24h] [rbp-6Ch]
  int v28; // [rsp+34h] [rbp-5Ch] BYREF
  __int64 v29; // [rsp+38h] [rbp-58h] BYREF
  _QWORD *v30; // [rsp+40h] [rbp-50h] BYREF
  _QWORD v31[8]; // [rsp+50h] [rbp-40h] BYREF

  sub_93FA70(&v30, a1, a3);
  v5 = (const char *)sub_93EC00((__int64)v30, a3);
  sub_93ED80(*(_DWORD *)(a1 + 448), (char *)&v28);
  v25 = sub_9405D0(a1, *(_DWORD *)(a1 + 448), v6, v7, v8, v9);
  v10 = sub_ADD430(a1 + 16, 0, 0);
  v11 = sub_ADCD40(a1 + 16, v10, 0, 0);
  v12 = 1;
  v24 = v11;
  if ( (*(_BYTE *)(a3 + 202) & 1) == 0 )
    v12 = (*(_BYTE *)(a2 + 32) & 0xF) == 7;
  v26 = sub_AF3490(v12, 1, unk_4D04660 != 0, 0, 0);
  v27 = v28;
  v13 = sub_BD5D20(a2);
  v15 = 0;
  v16 = v13;
  v17 = v14;
  if ( v5 )
  {
    v22 = v13;
    v23 = v14;
    v18 = strlen(v5);
    v16 = v22;
    v17 = v23;
    v15 = v18;
  }
  v19 = sub_ADE3D0(
          (int)a1 + 16,
          v25,
          (_DWORD)v5,
          v15,
          v16,
          v17,
          v25,
          v27,
          v24,
          v27,
          0,
          v26,
          0,
          0,
          0,
          0,
          (__int64)byte_3F871B3,
          0);
  v20 = *(_DWORD *)(a1 + 448);
  v29 = v19;
  if ( v20 && *(_WORD *)(a1 + 452) )
    sub_B994C0(a2, v19);
  sub_940270((__int64 *)(a1 + 464), &v29);
  result = v31;
  if ( v30 != v31 )
    return (_QWORD *)j_j___libc_free_0(v30, v31[0] + 1LL);
  return result;
}
