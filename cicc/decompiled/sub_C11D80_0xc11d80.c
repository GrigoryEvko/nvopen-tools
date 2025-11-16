// Function: sub_C11D80
// Address: 0xc11d80
//
_BYTE *__fastcall sub_C11D80(__int64 **a1, __int64 a2, char *a3)
{
  __int64 v3; // rbp
  __int64 *v4; // rax
  char v5; // dl
  __int64 v6; // rdi
  __int64 v7; // r8
  __int64 v8; // rcx
  unsigned int v9; // eax
  void *v11; // [rsp-48h] [rbp-48h] BYREF
  int v12; // [rsp-40h] [rbp-40h]
  char v13; // [rsp-3Ch] [rbp-3Ch]
  __int64 v14; // [rsp-38h] [rbp-38h]
  __int64 v15; // [rsp-30h] [rbp-30h]
  __int64 v16; // [rsp-28h] [rbp-28h]
  char v17; // [rsp-20h] [rbp-20h]
  __int64 v18; // [rsp-18h] [rbp-18h]
  __int64 v19; // [rsp-8h] [rbp-8h]

  v4 = *a1;
  v5 = *a3;
  v6 = **a1;
  v7 = v4[21];
  v8 = v4[22];
  v9 = *(_DWORD *)(a2 + 56);
  if ( v9 > 3 )
    BUG();
  v19 = v3;
  v13 = v9;
  v14 = a2;
  v12 = 25;
  v11 = &unk_49D9E20;
  v15 = v7;
  v16 = v8;
  v17 = v5;
  v18 = 0;
  return sub_B6EB20(v6, (__int64)&v11);
}
