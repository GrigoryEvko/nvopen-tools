// Function: sub_2EA9D30
// Address: 0x2ea9d30
//
_BYTE *__fastcall sub_2EA9D30(__int64 *a1, __int64 a2, char *a3, __int64 *a4, _QWORD *a5)
{
  char v6; // r13
  __int64 v7; // rdx
  __int64 v8; // rax
  __int64 v9; // rdi
  __int64 v10; // rsi
  __int64 v11; // rcx
  void *v13; // [rsp+0h] [rbp-60h] BYREF
  int v14; // [rsp+8h] [rbp-58h]
  char v15; // [rsp+Ch] [rbp-54h]
  __int64 v16; // [rsp+10h] [rbp-50h]
  __int64 v17; // [rsp+18h] [rbp-48h]
  __int64 v18; // [rsp+20h] [rbp-40h]
  char v19; // [rsp+28h] [rbp-38h]
  __int64 v20; // [rsp+30h] [rbp-30h]

  v6 = *a3;
  v7 = 0;
  if ( v6 )
    v7 = sub_2EA9C60(a2, a4, a5);
  v8 = a1[1];
  v9 = *a1;
  v10 = *(_QWORD *)(v8 + 168);
  v11 = *(_QWORD *)(v8 + 176);
  if ( *(_DWORD *)(a2 + 56) > 3u )
    BUG();
  v15 = *(_DWORD *)(a2 + 56);
  v17 = v10;
  v16 = a2;
  v19 = v6;
  v14 = 25;
  v13 = &unk_49D9E20;
  v18 = v11;
  v20 = v7;
  return sub_B6EB20(v9, (__int64)&v13);
}
