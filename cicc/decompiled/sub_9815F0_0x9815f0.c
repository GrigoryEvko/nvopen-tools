// Function: sub_9815F0
// Address: 0x9815f0
//
__int64 __fastcall sub_9815F0(__int64 a1, __int16 a2)
{
  _QWORD *v3; // r14
  unsigned __int64 v4; // r15
  _QWORD *v5; // rax
  _QWORD *v6; // r12
  __int64 v7; // rcx
  __int64 v8; // rdx
  _QWORD *v9; // r13
  __int64 v10; // rax
  int v11; // esi
  __int64 v12; // r14
  __int64 v13; // rcx
  __int64 v14; // rdx
  __int64 v15; // rax
  int v16; // ecx
  __int64 v17; // rsi
  __int64 v18; // rax
  __int64 v19; // rdx
  _BOOL8 v20; // rdi
  __int64 v22; // rax
  _QWORD *v23; // rax
  _QWORD *v24; // rdx
  _BOOL8 v25; // rdi
  __int64 v26; // rdi
  _QWORD *v27; // [rsp+8h] [rbp-48h]
  int v28; // [rsp+10h] [rbp-40h]
  _QWORD *v29; // [rsp+10h] [rbp-40h]
  _QWORD *v30; // [rsp+10h] [rbp-40h]

  v3 = (_QWORD *)sub_C52410();
  v4 = sub_C959E0();
  v5 = (_QWORD *)v3[2];
  v6 = v3 + 1;
  if ( !v5 )
    goto LABEL_21;
  do
  {
    while ( 1 )
    {
      v7 = v5[2];
      v8 = v5[3];
      if ( v4 <= v5[4] )
        break;
      v5 = (_QWORD *)v5[3];
      if ( !v8 )
        goto LABEL_6;
    }
    v6 = v5;
    v5 = (_QWORD *)v5[2];
  }
  while ( v7 );
LABEL_6:
  if ( v3 + 1 == v6 || (v9 = v6 + 6, v4 < v6[4]) )
  {
LABEL_21:
    v29 = v6;
    v27 = v3 + 1;
    v22 = sub_22077B0(88);
    v9 = (_QWORD *)(v22 + 48);
    *(_QWORD *)(v22 + 32) = v4;
    v6 = (_QWORD *)v22;
    *(_DWORD *)(v22 + 48) = 0;
    *(_QWORD *)(v22 + 56) = 0;
    *(_QWORD *)(v22 + 64) = v22 + 48;
    *(_QWORD *)(v22 + 72) = v22 + 48;
    *(_QWORD *)(v22 + 80) = 0;
    v23 = sub_981350(v3, v29, (unsigned __int64 *)(v22 + 32));
    if ( v24 )
    {
      v25 = v23 || v27 == v24 || v4 < v24[4];
      sub_220F040(v25, v6, v24, v27);
      ++v3[5];
      v10 = v6[7];
      if ( v10 )
        goto LABEL_9;
LABEL_26:
      v12 = (__int64)v9;
LABEL_15:
      v15 = sub_22077B0(40);
      v16 = *(_DWORD *)(a1 + 8);
      v17 = v12;
      *(_DWORD *)(v15 + 36) = 0;
      v12 = v15;
      *(_DWORD *)(v15 + 32) = v16;
      v28 = v16;
      v18 = sub_9814F0(v6 + 5, v17, (int *)(v15 + 32));
      if ( v19 )
      {
        v20 = v18 || v9 == (_QWORD *)v19 || v28 < *(_DWORD *)(v19 + 32);
        sub_220F040(v20, v12, v19, v9);
        ++v6[10];
      }
      else
      {
        v26 = v12;
        v12 = v18;
        j_j___libc_free_0(v26, 40);
      }
      goto LABEL_20;
    }
    v30 = v23;
    sub_97E950(0);
    j_j___libc_free_0(v6, 88);
    v9 = v30 + 6;
    v6 = v30;
  }
  v10 = v6[7];
  if ( !v10 )
    goto LABEL_26;
LABEL_9:
  v11 = *(_DWORD *)(a1 + 8);
  v12 = (__int64)v9;
  do
  {
    while ( 1 )
    {
      v13 = *(_QWORD *)(v10 + 16);
      v14 = *(_QWORD *)(v10 + 24);
      if ( *(_DWORD *)(v10 + 32) >= v11 )
        break;
      v10 = *(_QWORD *)(v10 + 24);
      if ( !v14 )
        goto LABEL_13;
    }
    v12 = v10;
    v10 = *(_QWORD *)(v10 + 16);
  }
  while ( v13 );
LABEL_13:
  if ( v9 == (_QWORD *)v12 || v11 < *(_DWORD *)(v12 + 32) )
    goto LABEL_15;
LABEL_20:
  --*(_DWORD *)(v12 + 36);
  *(_WORD *)(a1 + 14) = a2;
  return 0;
}
