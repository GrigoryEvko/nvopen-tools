// Function: sub_3147DF0
// Address: 0x3147df0
//
__int64 __fastcall sub_3147DF0(__int64 a1, char a2)
{
  __int64 v2; // rbx
  __int64 v3; // r12
  const char *v4; // rax
  unsigned __int64 v5; // rdx
  __int64 j; // rbx
  _QWORD *v7; // r12
  __int64 v8; // rdx
  __int64 v9; // rcx
  unsigned __int64 v10; // r8
  __int64 v11; // r9
  __int64 v12; // r13
  unsigned __int64 v13; // r12
  unsigned __int64 v14; // r12
  unsigned __int64 v15; // rdi
  _QWORD v17[6]; // [rsp+10h] [rbp-C0h] BYREF
  __int64 v18; // [rsp+40h] [rbp-90h] BYREF
  char v19; // [rsp+48h] [rbp-88h]
  _BYTE v20[16]; // [rsp+50h] [rbp-80h] BYREF
  void (__fastcall *v21)(_BYTE *, _BYTE *, __int64); // [rsp+60h] [rbp-70h]
  unsigned __int64 v22; // [rsp+70h] [rbp-60h]
  unsigned __int64 v23; // [rsp+78h] [rbp-58h]
  __int64 v24; // [rsp+80h] [rbp-50h]
  __int64 v25; // [rsp+88h] [rbp-48h]
  __int64 v26; // [rsp+90h] [rbp-40h]
  unsigned int i; // [rsp+98h] [rbp-38h]

  v2 = *(_QWORD *)(a1 + 16);
  v18 = 4;
  v19 = a2;
  v21 = 0;
  v22 = 0;
  v23 = 0;
  v24 = 0;
  v25 = 0;
  v26 = 0;
  for ( i = 0; a1 + 8 != v2; v2 = *(_QWORD *)(v2 + 8) )
  {
    v3 = v2 - 56;
    if ( !v2 )
      v3 = 0;
    if ( !sub_B2FC80(v3) )
    {
      v4 = sub_BD5D20(v3);
      if ( v5 <= 4 || *(_DWORD *)v4 != 1836477548 || v4[4] != 46 )
      {
        v17[1] = 23456;
        v17[0] = v18;
        v17[2] = *(unsigned __int8 *)(*(_QWORD *)(v3 + 24) + 8LL);
        v18 = sub_CBF760(v17, 0x18u);
      }
    }
  }
  for ( j = *(_QWORD *)(a1 + 32); a1 + 24 != j; j = *(_QWORD *)(j + 8) )
  {
    v7 = (_QWORD *)(j - 56);
    if ( !j )
      v7 = 0;
    if ( !sub_B2FC80((__int64)v7) )
      sub_3146A80(&v18, v7, v8, v9, v10, v11);
  }
  v12 = v18;
  sub_C7D6A0(v25, 16LL * i, 8);
  v13 = v23;
  if ( v23 )
  {
    sub_C7D6A0(*(_QWORD *)(v23 + 8), 16LL * *(unsigned int *)(v23 + 24), 8);
    j_j___libc_free_0(v13);
  }
  v14 = v22;
  if ( v22 )
  {
    v15 = *(_QWORD *)(v22 + 32);
    if ( v15 != v22 + 48 )
      _libc_free(v15);
    sub_C7D6A0(*(_QWORD *)(v14 + 8), 8LL * *(unsigned int *)(v14 + 24), 4);
    j_j___libc_free_0(v14);
  }
  if ( v21 )
    v21(v20, v20, 3);
  return v12;
}
