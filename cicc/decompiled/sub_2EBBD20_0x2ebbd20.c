// Function: sub_2EBBD20
// Address: 0x2ebbd20
//
void __fastcall sub_2EBBD20(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned __int64 v4; // rdi
  unsigned __int64 v6; // rdx
  char v7; // cl
  __int64 v8; // rbx
  __int64 v9; // rax
  __int64 v10; // r8
  __int64 v11; // rdx
  unsigned __int64 v12; // rcx
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // r8
  unsigned __int64 v16; // rsi
  unsigned __int64 v17; // rcx
  unsigned __int64 v18; // rcx
  char v19[8]; // [rsp+0h] [rbp-40h] BYREF
  __int64 v20; // [rsp+8h] [rbp-38h]
  __int64 v21; // [rsp+10h] [rbp-30h]
  unsigned __int64 v22; // [rsp+18h] [rbp-28h]

  v4 = *(unsigned int *)(a2 + 624);
  if ( !*(_DWORD *)(a2 + 624) )
    return;
  if ( v4 == 1 )
  {
    v13 = sub_2EB70C0(a2);
    if ( a3 )
    {
      v16 = *(unsigned int *)(a3 + 624);
      v19[0] = 0;
      v17 = v14 & 0xFFFFFFFFFFFFFFF8LL;
      v20 = a3;
      v22 = v16;
      v21 = a3;
      if ( (v14 & 4) != 0 )
        sub_2EBBA90(a1, (__int64)v19, v13, v17);
      else
        sub_2EBB8B0(a1, (__int64)v19, v13, v17, v15);
    }
    else
    {
      v18 = v14 & 0xFFFFFFFFFFFFFFF8LL;
      if ( (v14 & 4) != 0 )
        sub_2EBBA90(a1, 0, v13, v18);
      else
        sub_2EBB8B0(a1, 0, v13, v18, v15);
    }
    return;
  }
  v21 = a3;
  v6 = *(unsigned int *)(a1 + 56);
  v19[0] = 0;
  v20 = a2;
  v22 = v4;
  if ( v6 <= 0x64 )
  {
    v7 = 0;
    if ( v4 <= v6 )
      goto LABEL_5;
LABEL_13:
    sub_2EBA1B0(a1, (__int64)v19);
    v7 = v19[0];
    if ( !v22 )
      return;
    goto LABEL_5;
  }
  v7 = 0;
  if ( v6 / 0x28 < v4 )
    goto LABEL_13;
LABEL_5:
  v8 = 0;
  while ( !v7 )
  {
    v9 = sub_2EB70C0(v20);
    v12 = v11 & 0xFFFFFFFFFFFFFFF8LL;
    if ( (v11 & 4) != 0 )
    {
      sub_2EBBA90(a1, (__int64)v19, v9, v12);
      if ( v22 <= ++v8 )
        return;
    }
    else
    {
      sub_2EBB8B0(a1, (__int64)v19, v9, v12, v10);
      if ( v22 <= ++v8 )
        return;
    }
    v7 = v19[0];
  }
}
