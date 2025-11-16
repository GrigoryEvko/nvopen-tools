// Function: sub_177F430
// Address: 0x177f430
//
bool __fastcall sub_177F430(__int64 a1, __int64 a2, unsigned __int64 *a3, char a4)
{
  unsigned int v7; // r15d
  int v8; // r8d
  bool result; // al
  unsigned int v10; // edx
  __int64 v11; // rdi
  unsigned int v12; // ecx
  char v13; // dl
  unsigned __int64 *v14; // rdx
  __int64 v15; // rdi
  int v16; // eax
  int v17; // r8d
  unsigned __int64 v18; // r12
  unsigned int v19; // ebx
  int v20; // eax
  unsigned int v21; // [rsp+8h] [rbp-48h]
  unsigned int v22; // [rsp+Ch] [rbp-44h]
  unsigned int v23; // [rsp+Ch] [rbp-44h]
  bool v24; // [rsp+Ch] [rbp-44h]
  unsigned __int64 v25; // [rsp+10h] [rbp-40h] BYREF
  unsigned int v26; // [rsp+18h] [rbp-38h]

  v7 = *(_DWORD *)(a2 + 8);
  if ( v7 <= 0x40 )
  {
    result = 0;
    if ( !*(_QWORD *)a2 )
      return result;
    v10 = *(_DWORD *)(a1 + 8);
    if ( !a4 )
      goto LABEL_12;
  }
  else
  {
    v8 = sub_16A57B0(a2);
    result = 0;
    if ( v7 == v8 )
      return result;
    v10 = *(_DWORD *)(a1 + 8);
    if ( !a4 )
      goto LABEL_12;
  }
  v11 = *(_QWORD *)a1;
  v12 = v10 - 1;
  if ( v10 <= 0x40 )
  {
    if ( v11 != 1LL << v12 )
    {
      v26 = v10;
LABEL_13:
      v14 = a3;
      v15 = a1;
      v25 = 0;
      if ( a4 )
        goto LABEL_14;
LABEL_8:
      sub_16ADD10(v15, a2, v14, &v25);
      goto LABEL_15;
    }
  }
  else
  {
    v22 = v10 - 1;
    if ( (*(_QWORD *)(v11 + 8LL * (v12 >> 6)) & (1LL << v12)) == 0
      || (v21 = v10, v20 = sub_16A58A0(a1), v10 = v21, v20 != v22) )
    {
      v26 = v10;
      v13 = 1;
      goto LABEL_7;
    }
  }
  if ( v7 <= 0x40 )
  {
    result = 0;
    if ( *(_QWORD *)a2 == 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v7) )
      return result;
  }
  else
  {
    v23 = v10;
    v16 = sub_16A58F0(a2);
    v10 = v23;
    v17 = v16;
    result = 0;
    if ( v7 == v17 )
      return result;
  }
LABEL_12:
  v26 = v10;
  if ( v10 <= 0x40 )
    goto LABEL_13;
  v13 = a4;
LABEL_7:
  sub_16A4EF0((__int64)&v25, 0, v13);
  v14 = a3;
  v15 = a1;
  if ( !a4 )
    goto LABEL_8;
LABEL_14:
  sub_16AE5C0(v15, a2, (__int64)v14, (__int64)&v25);
LABEL_15:
  v18 = v25;
  v19 = v26;
  result = v25 == 0;
  if ( v26 > 0x40 )
  {
    result = v19 == (unsigned int)sub_16A57B0((__int64)&v25);
    if ( v18 )
    {
      v24 = result;
      j_j___libc_free_0_0(v18);
      return v24;
    }
  }
  return result;
}
