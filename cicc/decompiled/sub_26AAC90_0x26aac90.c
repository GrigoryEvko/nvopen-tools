// Function: sub_26AAC90
// Address: 0x26aac90
//
__int64 __fastcall sub_26AAC90(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rdi
  int v8; // eax
  __int64 v9; // rax
  __int64 v10; // r13
  __int64 v11; // r14
  __int64 v12; // rax
  __int64 v14; // r13
  __int64 *v15; // rbx
  __int64 v16; // r13
  __int64 *v17; // r14
  __int64 *v18; // rsi
  unsigned __int64 v19; // [rsp+0h] [rbp-50h]
  unsigned __int64 v20[2]; // [rsp+18h] [rbp-38h] BYREF
  _BYTE v21[40]; // [rsp+28h] [rbp-28h] BYREF

  v7 = *(_QWORD *)a2;
  v8 = *(_DWORD *)(a2 + 16);
  v20[0] = (unsigned __int64)v21;
  v20[1] = 0;
  if ( v8 )
    sub_266E590((__int64)v20, (char **)(a2 + 8), a3, a4, a5, a6);
  v9 = sub_B43CB0(v7);
  v10 = *a1;
  v11 = a1[1];
  v19 = v9 & 0xFFFFFFFFFFFFFFFCLL;
  nullsub_1518();
  v12 = sub_26A73D0(v10, v19, 0, v11, 0, 1);
  if ( v12 && *(_BYTE *)(v12 + 337) )
  {
    v14 = a1[1];
    v15 = *(__int64 **)(v12 + 376);
    v16 = v14 + 344;
    v17 = &v15[*(unsigned int *)(v12 + 384)];
    while ( v17 != v15 )
    {
      v18 = v15++;
      sub_2699F90(v16, v18);
    }
  }
  else
  {
    *(_BYTE *)(a1[1] + 337) = *(_BYTE *)(a1[1] + 336);
  }
  if ( (_BYTE *)v20[0] != v21 )
    _libc_free(v20[0]);
  return 1;
}
