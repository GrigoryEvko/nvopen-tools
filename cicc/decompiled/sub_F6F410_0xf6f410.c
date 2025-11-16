// Function: sub_F6F410
// Address: 0xf6f410
//
__int64 __fastcall sub_F6F410(__int64 a1, __int64 a2, __int64 *a3)
{
  __int64 v5; // rbx
  void *v6; // r15
  __int64 *v7; // r14
  __int64 v8; // rax
  __int64 v9; // r15
  int *v10; // rdx
  __int64 v11; // r11
  __int64 *v12; // rdi
  __int64 result; // rax
  _QWORD *v14; // rax
  __int64 v15; // rbx
  __int64 v16; // r14
  __int64 v17; // rdx
  unsigned int v18; // esi
  __int64 v19; // [rsp-10h] [rbp-C0h]
  void *v20; // [rsp+8h] [rbp-A8h]
  __int64 v21; // [rsp+10h] [rbp-A0h]
  __int64 v22; // [rsp+18h] [rbp-98h]
  const char *v23; // [rsp+20h] [rbp-90h] BYREF
  char v24; // [rsp+40h] [rbp-70h]
  char v25; // [rsp+41h] [rbp-6Fh]
  _QWORD v26[4]; // [rsp+50h] [rbp-60h] BYREF
  __int16 v27; // [rsp+70h] [rbp-40h]

  v5 = *a3;
  v6 = *(void **)a2;
  v23 = "rdx.shuf";
  v7 = *(__int64 **)a1;
  v8 = *(unsigned int *)(a2 + 8);
  v25 = 1;
  v24 = 3;
  v20 = v6;
  v22 = v8;
  v21 = sub_ACADE0(*(__int64 ***)(v5 + 8));
  v9 = (*(__int64 (__fastcall **)(__int64, __int64, __int64, void *, __int64))(*(_QWORD *)v7[10] + 112LL))(
         v7[10],
         v5,
         v21,
         v6,
         v22);
  if ( !v9 )
  {
    v27 = 257;
    v14 = sub_BD2C40(112, unk_3F1FE60);
    v9 = (__int64)v14;
    if ( v14 )
      sub_B4E9E0((__int64)v14, v5, v21, v20, v22, (__int64)v26, 0, 0);
    (*(void (__fastcall **)(__int64, __int64, const char **, __int64, __int64))(*(_QWORD *)v7[11] + 16LL))(
      v7[11],
      v9,
      &v23,
      v7[7],
      v7[8]);
    v15 = *v7;
    v16 = *v7 + 16LL * *((unsigned int *)v7 + 2);
    while ( v16 != v15 )
    {
      v17 = *(_QWORD *)(v15 + 8);
      v18 = *(_DWORD *)v15;
      v15 += 16;
      sub_B99FD0(v9, v18, v17);
    }
  }
  v10 = *(int **)(a1 + 8);
  v11 = *a3;
  v12 = *(__int64 **)a1;
  if ( (unsigned int)(*v10 - 53) <= 1 )
  {
    result = sub_F6F180((__int64)v12, **(_DWORD **)(a1 + 16), *a3, v9);
    *a3 = result;
  }
  else
  {
    v26[0] = "bin.rdx";
    v27 = 259;
    *a3 = sub_F6BB60(v12, *v10, v11, v9, (int)v23, 0, (__int64)v26, 0);
    return v19;
  }
  return result;
}
