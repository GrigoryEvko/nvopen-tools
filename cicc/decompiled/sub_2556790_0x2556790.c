// Function: sub_2556790
// Address: 0x2556790
//
void __fastcall sub_2556790(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // rdi
  int v9; // eax
  __int64 v10; // r14
  __int64 v11; // r8
  __int64 v12; // r9
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // r15
  __int64 v16; // rax
  unsigned __int8 v17; // di
  unsigned __int64 v18; // rax
  char *v19; // [rsp+8h] [rbp-68h] BYREF
  __int64 v20; // [rsp+10h] [rbp-60h]
  char v21; // [rsp+18h] [rbp-58h] BYREF
  __int64 v22; // [rsp+20h] [rbp-50h]
  unsigned __int64 v23[2]; // [rsp+28h] [rbp-48h] BYREF
  _BYTE v24[56]; // [rsp+38h] [rbp-38h] BYREF

  v8 = *(_QWORD *)a3;
  v9 = *(_DWORD *)(a3 + 16);
  v19 = &v21;
  v20 = 0;
  if ( v9 )
  {
    sub_2538240((__int64)&v19, (char **)(a3 + 8), a3, a4, a5, a6);
    v14 = (unsigned int)v20;
    v10 = *(unsigned int *)(*(_QWORD *)(a2 + 16) + 32LL);
    if ( (_DWORD)v20 )
      goto LABEL_6;
  }
  else
  {
    v10 = *(unsigned int *)(*(_QWORD *)(a2 + 16) + 32LL);
  }
  if ( !sub_B491E0(v8) )
  {
    v13 = v8;
    v14 = (unsigned int)v20;
    v15 = *(_QWORD *)(v8 + 32 * (v10 - (*(_DWORD *)(v8 + 4) & 0x7FFFFFF)));
    goto LABEL_8;
  }
  v14 = (unsigned int)v20;
LABEL_6:
  v15 = 0;
  v16 = *(int *)&v19[4 * (unsigned int)(v10 + 1)];
  v13 = v8;
  if ( (int)v16 >= 0 )
    v15 = *(_QWORD *)(v8 + 32 * (v16 - (*(_DWORD *)(v8 + 4) & 0x7FFFFFF)));
LABEL_8:
  v22 = v13;
  v23[0] = (unsigned __int64)v24;
  v23[1] = 0;
  if ( (_DWORD)v14 )
  {
    sub_2538550((__int64)v23, (__int64)&v19, v13, v14, v11, v12);
    v13 = v22;
  }
  v17 = -1;
  if ( *a1 )
  {
    v18 = *(_QWORD *)(*a1 + 104LL);
    if ( v18 )
    {
      _BitScanReverse64(&v18, v18);
      v17 = 63 - (v18 ^ 0x3F);
    }
  }
  sub_25562B0(v17, *(_QWORD *)(a1[1] + 104LL), v13, v15, a4);
  if ( (_BYTE *)v23[0] != v24 )
    _libc_free(v23[0]);
  if ( v19 != &v21 )
    _libc_free((unsigned __int64)v19);
}
