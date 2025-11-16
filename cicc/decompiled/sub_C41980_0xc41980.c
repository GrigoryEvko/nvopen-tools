// Function: sub_C41980
// Address: 0xc41980
//
__int64 __fastcall sub_C41980(void **a1, __int64 a2, char a3, _BYTE *a4)
{
  __int64 v6; // rdx
  _QWORD *v7; // rsi
  unsigned int v8; // r14d
  unsigned __int64 v9; // rdx
  _QWORD *v10; // rax
  _QWORD *v11; // rcx
  unsigned int v12; // r13d
  _QWORD *v13; // rdi
  __int64 v15; // [rsp+8h] [rbp-88h]
  unsigned __int8 v16; // [rsp+10h] [rbp-80h]
  unsigned __int64 v17; // [rsp+10h] [rbp-80h]
  __int64 v19; // [rsp+20h] [rbp-70h] BYREF
  int v20; // [rsp+28h] [rbp-68h]
  _QWORD *v21; // [rsp+30h] [rbp-60h] BYREF
  __int64 v22; // [rsp+38h] [rbp-58h]
  _QWORD v23[10]; // [rsp+40h] [rbp-50h] BYREF

  v6 = *(unsigned int *)(a2 + 8);
  v7 = v23;
  v21 = v23;
  v8 = v6;
  v22 = 0x400000000LL;
  v9 = (unsigned __int64)(v6 + 63) >> 6;
  if ( !v9 )
    goto LABEL_9;
  v10 = v23;
  if ( v9 > 4 )
  {
    v17 = v9;
    sub_C8D5F0(&v21, v23, v9, 8);
    v7 = v21;
    v9 = v17;
    v10 = &v21[(unsigned int)v22];
    v11 = &v21[v17];
    if ( v11 != v10 )
      goto LABEL_4;
  }
  else
  {
    v11 = &v23[v9];
    if ( v11 != v23 )
    {
      do
      {
LABEL_4:
        if ( v10 )
          *v10 = 0;
        ++v10;
      }
      while ( v11 != v10 );
      v7 = v21;
    }
  }
  LODWORD(v22) = v9;
LABEL_9:
  v15 = v9;
  v16 = *(_BYTE *)(a2 + 12) ^ 1;
  if ( *a1 == sub_C33340() )
    v12 = sub_C3FF40((__int64)a1, v7, v15, v8, v16, a3, a4);
  else
    v12 = sub_C34710((__int64)a1, v7, v15, v8, v16, a3, a4);
  sub_C438C0(&v19, v8, v21, (unsigned int)v22);
  if ( *(_DWORD *)(a2 + 8) > 0x40u && *(_QWORD *)a2 )
    j_j___libc_free_0_0(*(_QWORD *)a2);
  v13 = v21;
  *(_QWORD *)a2 = v19;
  *(_DWORD *)(a2 + 8) = v20;
  if ( v13 != v23 )
    _libc_free(v13, v8);
  return v12;
}
