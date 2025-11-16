// Function: sub_1CC3DA0
// Address: 0x1cc3da0
//
__int64 __fastcall sub_1CC3DA0(int *a1, unsigned int a2)
{
  unsigned __int64 v3; // r14
  _QWORD *v4; // rax
  _DWORD *v5; // r12
  __int64 v6; // rcx
  __int64 v7; // rdx
  _DWORD *v8; // r15
  __int64 v9; // rax
  __int64 v10; // r14
  int v11; // esi
  __int64 v12; // rcx
  __int64 v13; // rdx
  int v14; // ebx
  __int64 v15; // rax
  _DWORD *v16; // rdx
  _BOOL8 v17; // rdi
  __int64 v19; // rax
  _QWORD *v20; // rax
  __int64 v21; // rdx
  _BOOL8 v22; // rdi
  __int64 v23; // [rsp+0h] [rbp-40h]
  _QWORD *v24; // [rsp+0h] [rbp-40h]
  _DWORD *v25; // [rsp+0h] [rbp-40h]
  __int64 v27; // [rsp+8h] [rbp-38h]

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
    v24 = v5;
    v19 = sub_22077B0(88);
    v8 = (_DWORD *)(v19 + 48);
    *(_QWORD *)(v19 + 32) = v3;
    v5 = (_DWORD *)v19;
    *(_DWORD *)(v19 + 48) = 0;
    *(_QWORD *)(v19 + 56) = 0;
    *(_QWORD *)(v19 + 64) = v19 + 48;
    *(_QWORD *)(v19 + 72) = v19 + 48;
    *(_QWORD *)(v19 + 80) = 0;
    v20 = sub_981350(&qword_4FA0200, v24, (unsigned __int64 *)(v19 + 32));
    if ( v21 )
    {
      v22 = v20 || dword_4FA0208 == (_DWORD *)v21 || v3 < *(_QWORD *)(v21 + 32);
      sub_220F040(v22, v5, v21, dword_4FA0208);
      v9 = *((_QWORD *)v5 + 7);
      ++*(_QWORD *)&dword_4FA0208[8];
      if ( v9 )
        goto LABEL_9;
LABEL_26:
      v10 = (__int64)v8;
LABEL_15:
      v23 = v10;
      v10 = sub_22077B0(40);
      *(_DWORD *)(v10 + 36) = 0;
      v14 = *a1;
      *(_DWORD *)(v10 + 32) = *a1;
      v15 = sub_9814F0((_QWORD *)v5 + 5, v23, (int *)(v10 + 32));
      if ( v16 )
      {
        v17 = v8 == v16 || v15 || v14 < v16[8];
        sub_220F040(v17, v10, v16, v8);
        ++*((_QWORD *)v5 + 10);
      }
      else
      {
        v27 = v15;
        j_j___libc_free_0(v10, 40);
        v10 = v27;
      }
      goto LABEL_20;
    }
    v25 = v20;
    sub_1CC36E0(0);
    j_j___libc_free_0(v5, 88);
    v8 = v25 + 12;
    v5 = v25;
  }
  v9 = *((_QWORD *)v5 + 7);
  if ( !v9 )
    goto LABEL_26;
LABEL_9:
  v10 = (__int64)v8;
  v11 = *a1;
  do
  {
    while ( 1 )
    {
      v12 = *(_QWORD *)(v9 + 16);
      v13 = *(_QWORD *)(v9 + 24);
      if ( *(_DWORD *)(v9 + 32) >= v11 )
        break;
      v9 = *(_QWORD *)(v9 + 24);
      if ( !v13 )
        goto LABEL_13;
    }
    v10 = v9;
    v9 = *(_QWORD *)(v9 + 16);
  }
  while ( v12 );
LABEL_13:
  if ( (_DWORD *)v10 == v8 || v11 < *(_DWORD *)(v10 + 32) )
    goto LABEL_15;
LABEL_20:
  *(_DWORD *)(v10 + 36) = a2;
  return a2;
}
