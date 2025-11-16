// Function: sub_299DBB0
// Address: 0x299dbb0
//
__int64 __fastcall sub_299DBB0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v5; // r15
  __int64 *v9; // r13
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 v14; // r9
  __int64 *v15; // rbx
  __int64 *v16; // r12
  __int64 v17; // rdi
  __int64 v18; // [rsp+8h] [rbp-68h]
  __int64 *v19; // [rsp+10h] [rbp-60h] BYREF
  int v20; // [rsp+18h] [rbp-58h]
  char v21; // [rsp+20h] [rbp-50h] BYREF

  v5 = a1 + 32;
  v18 = a1 + 80;
  if ( (unsigned __int8)sub_B2D610(a3, 48) )
  {
    *(_QWORD *)(a1 + 8) = v5;
    *(_QWORD *)(a1 + 48) = 0;
    *(_QWORD *)(a1 + 56) = a1 + 80;
  }
  else
  {
    v9 = (__int64 *)(sub_BC1CD0(a4, &unk_4F8FAE8, a3) + 8);
    v10 = sub_BC1CD0(a4, &unk_4F875F0, a3);
    sub_D47CF0(&v19, v10 + 8);
    v15 = v19;
    v16 = &v19[v20];
    if ( v19 != v16 )
    {
      do
      {
        v17 = *v15++;
        sub_299D4D0(v17, v9, v11, v12, v13, v14);
      }
      while ( v16 != v15 );
      v16 = v19;
    }
    if ( v16 != (__int64 *)&v21 )
      _libc_free((unsigned __int64)v16);
    *(_QWORD *)(a1 + 8) = v5;
    *(_QWORD *)(a1 + 48) = 0;
    *(_QWORD *)(a1 + 56) = v18;
  }
  *(_QWORD *)(a1 + 64) = 2;
  *(_QWORD *)(a1 + 16) = 0x100000002LL;
  *(_DWORD *)(a1 + 72) = 0;
  *(_QWORD *)(a1 + 32) = &qword_4F82400;
  *(_BYTE *)(a1 + 76) = 1;
  *(_DWORD *)(a1 + 24) = 0;
  *(_BYTE *)(a1 + 28) = 1;
  *(_QWORD *)a1 = 1;
  return a1;
}
