// Function: sub_2FD5880
// Address: 0x2fd5880
//
__int64 __fastcall sub_2FD5880(__int64 a1, unsigned __int64 *a2, _QWORD *a3, __int64 a4)
{
  int v7; // r15d
  __int64 v8; // rax
  __int64 v9; // rax
  int v10; // r9d
  __int64 v11; // r14
  __int64 v12; // rax
  unsigned __int64 v13; // r8
  unsigned __int64 v14; // r14
  _QWORD *v15; // rsi
  char v16; // al
  __int64 v17; // rdx
  char v18; // r13
  __int64 v19; // rsi
  __int64 v20; // rbx
  __int64 v21; // r13
  unsigned __int64 v22; // rdi
  __int64 v24; // [rsp-8h] [rbp-F8h]
  int v25; // [rsp+8h] [rbp-E8h]
  _BYTE v26[64]; // [rsp+10h] [rbp-E0h] BYREF
  _BYTE *v27; // [rsp+50h] [rbp-A0h]
  __int64 v28; // [rsp+58h] [rbp-98h]
  _BYTE v29[64]; // [rsp+60h] [rbp-90h] BYREF
  __int64 v30; // [rsp+A0h] [rbp-50h]
  __int64 v31; // [rsp+A8h] [rbp-48h]
  __int64 v32; // [rsp+B0h] [rbp-40h]
  unsigned int v33; // [rsp+B8h] [rbp-38h]

  a3[43] &= 0xFFDuLL;
  v7 = sub_2EB2140(a4, &qword_501F1C0, (__int64)a3) + 8;
  v8 = sub_2EB2140(a4, &qword_50209B8, (__int64)a3);
  v9 = sub_2FD5470(v8 + 8, *(_QWORD *)(*a3 + 40LL));
  v10 = v9;
  if ( v9 && *(_QWORD *)(v9 + 8) )
  {
    v25 = v9;
    v11 = sub_2EB2140(a4, (__int64 *)&unk_501EC10, (__int64)a3) + 8;
    v12 = sub_22077B0(0x28u);
    v10 = v25;
    LODWORD(v13) = v12;
    if ( v12 )
    {
      *(_QWORD *)v12 = v11;
      *(_QWORD *)(v12 + 8) = 0;
      *(_QWORD *)(v12 + 16) = 0;
      *(_QWORD *)(v12 + 24) = 0;
      *(_DWORD *)(v12 + 32) = 0;
    }
    v14 = *a2;
    *a2 = v12;
    if ( v14 )
    {
      sub_C7D6A0(*(_QWORD *)(v14 + 16), 16LL * *(unsigned int *)(v14 + 32), 8);
      j_j___libc_free_0(v14);
      v13 = *a2;
      v10 = v25;
    }
    v30 = 0;
    v27 = v29;
    v28 = 0x1000000000LL;
    v31 = 0;
    v32 = 0;
    v33 = 0;
  }
  else
  {
    v30 = 0;
    LODWORD(v13) = 0;
    v27 = v29;
    v28 = 0x1000000000LL;
    v31 = 0;
    v32 = 0;
    v33 = 0;
  }
  v15 = a3;
  sub_2FD5DC0((unsigned int)v26, (_DWORD)a3, 1, v7, v13, v10, 0, 0);
  v16 = 0;
  v17 = v24;
  do
  {
    v18 = v16;
    v16 = sub_2FDBA40(v26, v15, v17);
  }
  while ( v16 );
  if ( v18 )
  {
    sub_2EAFFB0(a1);
  }
  else
  {
    *(_BYTE *)(a1 + 76) = 1;
    *(_QWORD *)(a1 + 8) = a1 + 32;
    *(_QWORD *)(a1 + 56) = a1 + 80;
    *(_QWORD *)(a1 + 16) = 0x100000002LL;
    *(_QWORD *)(a1 + 48) = 0;
    *(_QWORD *)(a1 + 64) = 2;
    *(_DWORD *)(a1 + 72) = 0;
    *(_DWORD *)(a1 + 24) = 0;
    *(_BYTE *)(a1 + 28) = 1;
    *(_QWORD *)(a1 + 32) = &qword_4F82400;
    *(_QWORD *)a1 = 1;
  }
  v19 = v33;
  if ( v33 )
  {
    v20 = v31;
    v21 = v31 + 32LL * v33;
    do
    {
      while ( 1 )
      {
        if ( *(_DWORD *)v20 <= 0xFFFFFFFD )
        {
          v22 = *(_QWORD *)(v20 + 8);
          if ( v22 )
            break;
        }
        v20 += 32;
        if ( v21 == v20 )
          goto LABEL_18;
      }
      v20 += 32;
      j_j___libc_free_0(v22);
    }
    while ( v21 != v20 );
LABEL_18:
    v19 = v33;
  }
  sub_C7D6A0(v31, 32 * v19, 8);
  if ( v27 != v29 )
    _libc_free((unsigned __int64)v27);
  return a1;
}
