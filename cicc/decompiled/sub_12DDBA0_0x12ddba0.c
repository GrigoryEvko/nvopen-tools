// Function: sub_12DDBA0
// Address: 0x12ddba0
//
__int64 __fastcall sub_12DDBA0(__int64 a1, int a2)
{
  unsigned __int64 v3; // r14
  _QWORD *v4; // rax
  _DWORD *v5; // r12
  __int64 v6; // rcx
  __int64 v7; // rdx
  _DWORD *v8; // r15
  __int64 v9; // rax
  int v10; // esi
  __int64 v11; // r14
  __int64 v12; // rcx
  __int64 v13; // rdx
  __int64 v14; // rax
  int v15; // ecx
  __int64 v16; // rax
  _DWORD *v17; // rdx
  _BOOL8 v18; // rdi
  __int64 v20; // rax
  _QWORD *v21; // rax
  __int64 v22; // rdx
  _BOOL8 v23; // rdi
  __int64 v24; // [rsp+0h] [rbp-40h]
  _QWORD *v25; // [rsp+0h] [rbp-40h]
  __int64 v26; // [rsp+0h] [rbp-40h]
  _DWORD *v27; // [rsp+0h] [rbp-40h]
  int v28; // [rsp+8h] [rbp-38h]

  v3 = sub_16D5D50();
  v4 = *(_QWORD **)&dword_4FA0208[2];
  v5 = dword_4FA0208;
  if ( !*(_QWORD *)&dword_4FA0208[2] )
    goto LABEL_21;
  do
  {
    while ( 1 )
    {
      v6 = v4[2];
      v7 = v4[3];
      if ( v3 <= v4[4] )
        break;
      v4 = (_QWORD *)v4[3];
      if ( !v7 )
        goto LABEL_6;
    }
    v5 = v4;
    v4 = (_QWORD *)v4[2];
  }
  while ( v6 );
LABEL_6:
  if ( v5 == dword_4FA0208 || (v8 = v5 + 12, v3 < *((_QWORD *)v5 + 4)) )
  {
LABEL_21:
    v25 = v5;
    v20 = sub_22077B0(88);
    v8 = (_DWORD *)(v20 + 48);
    *(_QWORD *)(v20 + 32) = v3;
    v5 = (_DWORD *)v20;
    *(_DWORD *)(v20 + 48) = 0;
    *(_QWORD *)(v20 + 56) = 0;
    *(_QWORD *)(v20 + 64) = v20 + 48;
    *(_QWORD *)(v20 + 72) = v20 + 48;
    *(_QWORD *)(v20 + 80) = 0;
    v21 = sub_981350(&qword_4FA0200, v25, (unsigned __int64 *)(v20 + 32));
    if ( v22 )
    {
      v23 = v21 || dword_4FA0208 == (_DWORD *)v22 || v3 < *(_QWORD *)(v22 + 32);
      sub_220F040(v23, v5, v22, dword_4FA0208);
      v9 = *((_QWORD *)v5 + 7);
      ++*(_QWORD *)&dword_4FA0208[8];
      if ( v9 )
        goto LABEL_9;
LABEL_26:
      v11 = (__int64)v8;
LABEL_15:
      v24 = v11;
      v14 = sub_22077B0(40);
      v15 = *(_DWORD *)(a1 + 8);
      *(_DWORD *)(v14 + 36) = 0;
      v11 = v14;
      *(_DWORD *)(v14 + 32) = v15;
      v28 = v15;
      v16 = sub_9814F0((_QWORD *)v5 + 5, v24, (int *)(v14 + 32));
      if ( v17 )
      {
        v18 = v16 || v8 == v17 || v28 < v17[8];
        sub_220F040(v18, v11, v17, v8);
        ++*((_QWORD *)v5 + 10);
      }
      else
      {
        v26 = v16;
        j_j___libc_free_0(v11, 40);
        v11 = v26;
      }
      goto LABEL_20;
    }
    v27 = v21;
    sub_12D4390(0);
    j_j___libc_free_0(v5, 88);
    v8 = v27 + 12;
    v5 = v27;
  }
  v9 = *((_QWORD *)v5 + 7);
  if ( !v9 )
    goto LABEL_26;
LABEL_9:
  v10 = *(_DWORD *)(a1 + 8);
  v11 = (__int64)v8;
  do
  {
    while ( 1 )
    {
      v12 = *(_QWORD *)(v9 + 16);
      v13 = *(_QWORD *)(v9 + 24);
      if ( *(_DWORD *)(v9 + 32) >= v10 )
        break;
      v9 = *(_QWORD *)(v9 + 24);
      if ( !v13 )
        goto LABEL_13;
    }
    v11 = v9;
    v9 = *(_QWORD *)(v9 + 16);
  }
  while ( v12 );
LABEL_13:
  if ( v8 == (_DWORD *)v11 || v10 < *(_DWORD *)(v11 + 32) )
    goto LABEL_15;
LABEL_20:
  --*(_DWORD *)(v11 + 36);
  *(_DWORD *)(a1 + 16) = a2;
  return 0;
}
