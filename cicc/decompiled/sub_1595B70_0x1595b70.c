// Function: sub_1595B70
// Address: 0x1595b70
//
__int64 __fastcall sub_1595B70(__int64 a1, __int64 a2, unsigned int a3)
{
  __int64 v5; // rsi
  __int64 v6; // rdi
  unsigned int *v7; // rbx
  char v8; // al
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // r12
  __int64 v16; // rax
  __int64 v17; // rdi
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 v21; // [rsp+0h] [rbp-30h] BYREF
  unsigned int v22; // [rsp+8h] [rbp-28h]

  v5 = a3;
  v6 = a2;
  v7 = (unsigned int *)sub_1595950(a2, a3);
  v8 = *(_BYTE *)(sub_1595890(a2) + 8);
  if ( v8 == 2 )
  {
    v20 = *v7;
    v22 = 32;
    v21 = v20;
    v12 = sub_1698270(a2, v5);
  }
  else if ( v8 == 3 )
  {
    v11 = *(_QWORD *)v7;
    v22 = 64;
    v21 = v11;
    v12 = sub_1698280(a2);
  }
  else
  {
    v19 = *(unsigned __int16 *)v7;
    v22 = 16;
    v21 = v19;
    v12 = sub_1698260(a2, v5, v9, v10);
  }
  v15 = v12;
  v16 = sub_16982C0(v6, v5, v13, v14);
  v17 = a1 + 8;
  if ( v15 == v16 )
    sub_169D060(v17, v15, &v21);
  else
    sub_169D050(v17, v15, &v21);
  if ( v22 > 0x40 && v21 )
    j_j___libc_free_0_0(v21);
  return a1;
}
