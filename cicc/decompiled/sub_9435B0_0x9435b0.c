// Function: sub_9435B0
// Address: 0x9435b0
//
_QWORD *__fastcall sub_9435B0(__int64 a1, __int64 a2, __int64 a3)
{
  const char *v5; // r14
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // r8
  int v9; // r9d
  __int64 v10; // rax
  _BOOL8 v11; // rdi
  int v12; // eax
  int v13; // edx
  int v14; // ecx
  int v15; // r8d
  int v16; // r9d
  int v17; // eax
  __int64 v18; // rax
  int v19; // edx
  _QWORD *result; // rax
  int v21; // [rsp+0h] [rbp-90h]
  int v22; // [rsp+8h] [rbp-88h]
  __int64 v23; // [rsp+10h] [rbp-80h]
  int v24; // [rsp+18h] [rbp-78h]
  int v25; // [rsp+1Ch] [rbp-74h]
  __int64 v26; // [rsp+28h] [rbp-68h]
  int v27; // [rsp+34h] [rbp-5Ch] BYREF
  __int64 v28; // [rsp+38h] [rbp-58h] BYREF
  _QWORD *v29; // [rsp+40h] [rbp-50h] BYREF
  _QWORD v30[8]; // [rsp+50h] [rbp-40h] BYREF

  sub_93FA70(&v29, a1, a3);
  v5 = (const char *)sub_93EC00((__int64)v29, a3);
  sub_93ED80(*(_DWORD *)(a1 + 448), (char *)&v27);
  v23 = sub_9405D0(a1, *(_DWORD *)(a1 + 448), v6, v7, v8, v9);
  v10 = sub_941B90(a1, *(_QWORD *)(a3 + 152));
  v11 = 1;
  v26 = v10;
  if ( (*(_BYTE *)(a3 + 202) & 1) == 0 )
    v11 = (*(_BYTE *)(a2 + 32) & 0xF) == 7;
  v24 = sub_AF3490(v11, 1, unk_4D04660 != 0, 0, 0);
  v25 = v27;
  v12 = sub_BD5D20(a2);
  v14 = 0;
  v15 = v12;
  v16 = v13;
  if ( v5 )
  {
    v21 = v12;
    v22 = v13;
    v17 = strlen(v5);
    v15 = v21;
    v16 = v22;
    v14 = v17;
  }
  v18 = sub_ADE3D0(
          (int)a1 + 16,
          v23,
          (_DWORD)v5,
          v14,
          v15,
          v16,
          v23,
          v25,
          v26,
          v25,
          0,
          v24,
          0,
          0,
          0,
          0,
          (__int64)byte_3F871B3,
          0);
  v19 = *(_DWORD *)(a1 + 448);
  v28 = v18;
  if ( v19 && *(_WORD *)(a1 + 452) )
    sub_B994C0(a2, v18);
  sub_940270((__int64 *)(a1 + 464), &v28);
  result = v30;
  if ( v29 != v30 )
    return (_QWORD *)j_j___libc_free_0(v29, v30[0] + 1LL);
  return result;
}
