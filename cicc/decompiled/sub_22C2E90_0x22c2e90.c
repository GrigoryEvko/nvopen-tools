// Function: sub_22C2E90
// Address: 0x22c2e90
//
void __fastcall sub_22C2E90(__int64 a1)
{
  __int64 v2; // rax
  unsigned __int64 v3; // r14
  unsigned __int64 v4; // rdi
  __int64 v5; // rax
  __int64 v6; // rax
  __int64 v7; // rbx
  __int64 v8; // r13
  __int64 i; // rdx
  __int64 v10; // rax
  __int64 v11; // rax
  _QWORD *v12; // rbx
  _QWORD *v13; // r15
  __int64 v14; // rax
  __int64 v15; // [rsp+18h] [rbp-88h] BYREF
  __int64 v16; // [rsp+20h] [rbp-80h]
  __int64 v17; // [rsp+28h] [rbp-78h]
  __int64 v18; // [rsp+30h] [rbp-70h]
  __int64 (__fastcall **v19)(); // [rsp+40h] [rbp-60h]
  __int64 v20; // [rsp+48h] [rbp-58h] BYREF
  __int64 v21; // [rsp+50h] [rbp-50h]
  __int64 v22; // [rsp+58h] [rbp-48h]
  __int64 v23; // [rsp+60h] [rbp-40h]

  v2 = sub_22C1580(a1);
  if ( !v2 )
    return;
  v3 = v2;
  sub_C7D6A0(*(_QWORD *)(v2 + 216), 16LL * *(unsigned int *)(v2 + 232), 8);
  v4 = *(_QWORD *)(v3 + 64);
  if ( v4 != v3 + 80 )
    _libc_free(v4);
  v5 = *(unsigned int *)(v3 + 56);
  if ( (_DWORD)v5 )
  {
    v12 = *(_QWORD **)(v3 + 40);
    v15 = 2;
    v16 = 0;
    v13 = &v12[5 * v5];
    v17 = -4096;
    v18 = 0;
    v20 = 2;
    v21 = 0;
    v22 = -8192;
    v19 = off_4A09D90;
    v23 = 0;
    do
    {
      v14 = v12[3];
      *v12 = &unk_49DB368;
      if ( v14 != 0 && v14 != -4096 && v14 != -8192 )
        sub_BD60C0(v12 + 1);
      v12 += 5;
    }
    while ( v13 != v12 );
    v19 = (__int64 (__fastcall **)())&unk_49DB368;
    sub_D68D70(&v20);
    sub_D68D70(&v15);
    v5 = *(unsigned int *)(v3 + 56);
  }
  sub_C7D6A0(*(_QWORD *)(v3 + 40), 40 * v5, 8);
  v6 = *(unsigned int *)(v3 + 24);
  if ( !(_DWORD)v6 )
    goto LABEL_6;
  v16 = 0;
  v15 = 2;
  LOBYTE(v18) = 0;
  v17 = -4096;
  v20 = 2;
  v21 = 0;
  v19 = (__int64 (__fastcall **)())&unk_49DE8C0;
  LOBYTE(v23) = 0;
  v22 = -8192;
  v7 = *(_QWORD *)(v3 + 8);
  v8 = v7 + 48LL * *(unsigned int *)(v3 + 24);
  if ( v7 == v8 )
    goto LABEL_22;
  for ( i = -4096; ; i = v17 )
  {
    v11 = *(_QWORD *)(v7 + 24);
    if ( v11 != i && v11 != v22 )
      sub_22C1BD0((unsigned __int64 *)(v7 + 40));
    if ( !*(_BYTE *)(v7 + 32) )
      break;
    *(_QWORD *)(v7 + 24) = 0;
    v7 += 48;
    *(_QWORD *)(v7 - 48) = &unk_49DB368;
    if ( v8 == v7 )
      goto LABEL_20;
LABEL_14:
    ;
  }
  v10 = *(_QWORD *)(v7 + 24);
  *(_QWORD *)v7 = &unk_49DB368;
  if ( v10 != -8192 && v10 != -4096 && v10 )
    sub_BD60C0((_QWORD *)(v7 + 8));
  v7 += 48;
  if ( v8 != v7 )
    goto LABEL_14;
LABEL_20:
  if ( (_BYTE)v23 )
    v22 = 0;
LABEL_22:
  v19 = (__int64 (__fastcall **)())&unk_49DB368;
  sub_D68D70(&v20);
  if ( (_BYTE)v18 )
    v17 = 0;
  sub_D68D70(&v15);
  v6 = *(unsigned int *)(v3 + 24);
LABEL_6:
  sub_C7D6A0(*(_QWORD *)(v3 + 8), 48 * v6, 8);
  j_j___libc_free_0(v3);
  *(_QWORD *)(a1 + 16) = 0;
}
