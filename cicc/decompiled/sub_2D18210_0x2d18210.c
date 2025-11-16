// Function: sub_2D18210
// Address: 0x2d18210
//
__int64 __fastcall sub_2D18210(int *a1, unsigned int a2)
{
  _QWORD *v4; // r14
  unsigned __int64 v5; // r15
  _QWORD *v6; // rax
  _QWORD *v7; // r12
  __int64 v8; // rcx
  __int64 v9; // rdx
  _QWORD *v10; // r8
  __int64 v11; // rax
  int v12; // esi
  __int64 v13; // r14
  __int64 v14; // rcx
  __int64 v15; // rdx
  __int64 v16; // rax
  int v17; // r15d
  __int64 v18; // rax
  __int64 v19; // rdx
  char v20; // di
  __int64 v22; // rax
  _QWORD *v23; // rax
  _QWORD *v24; // rdx
  char v25; // di
  unsigned __int64 v26; // rdi
  _QWORD *v27; // [rsp+8h] [rbp-48h]
  _QWORD *v28; // [rsp+10h] [rbp-40h]
  __int64 v29; // [rsp+10h] [rbp-40h]
  __int64 v30; // [rsp+18h] [rbp-38h]
  _QWORD *v31; // [rsp+18h] [rbp-38h]
  _QWORD *v32; // [rsp+18h] [rbp-38h]

  v4 = sub_C52410();
  v5 = sub_C959E0();
  v6 = (_QWORD *)v4[2];
  v7 = v4 + 1;
  if ( !v6 )
    goto LABEL_21;
  do
  {
    while ( 1 )
    {
      v8 = v6[2];
      v9 = v6[3];
      if ( v5 <= v6[4] )
        break;
      v6 = (_QWORD *)v6[3];
      if ( !v9 )
        goto LABEL_6;
    }
    v7 = v6;
    v6 = (_QWORD *)v6[2];
  }
  while ( v8 );
LABEL_6:
  if ( v4 + 1 == v7 || (v10 = v7 + 6, v5 < v7[4]) )
  {
LABEL_21:
    v31 = v7;
    v27 = v4 + 1;
    v22 = sub_22077B0(0x58u);
    *(_QWORD *)(v22 + 32) = v5;
    v7 = (_QWORD *)v22;
    *(_DWORD *)(v22 + 48) = 0;
    *(_QWORD *)(v22 + 56) = 0;
    *(_QWORD *)(v22 + 64) = v22 + 48;
    *(_QWORD *)(v22 + 72) = v22 + 48;
    *(_QWORD *)(v22 + 80) = 0;
    v29 = v22 + 48;
    v23 = sub_981350(v4, v31, (unsigned __int64 *)(v22 + 32));
    if ( v24 )
    {
      v25 = v23 || v27 == v24 || v5 < v24[4];
      sub_220F040(v25, (__int64)v7, v24, v27);
      ++v4[5];
      v11 = v7[7];
      v10 = (_QWORD *)v29;
      if ( v11 )
        goto LABEL_9;
LABEL_26:
      v13 = (__int64)v10;
LABEL_15:
      v28 = v10;
      v30 = v13;
      v16 = sub_22077B0(0x28u);
      v17 = *a1;
      *(_DWORD *)(v16 + 36) = 0;
      v13 = v16;
      *(_DWORD *)(v16 + 32) = v17;
      v18 = sub_9814F0(v7 + 5, v30, (int *)(v16 + 32));
      if ( v19 )
      {
        v20 = v28 == (_QWORD *)v19 || v18 || v17 < *(_DWORD *)(v19 + 32);
        sub_220F040(v20, v13, (_QWORD *)v19, v28);
        ++v7[10];
      }
      else
      {
        v26 = v13;
        v13 = v18;
        j_j___libc_free_0(v26);
      }
      goto LABEL_20;
    }
    v32 = v23;
    sub_2D17C50(0);
    j_j___libc_free_0((unsigned __int64)v7);
    v10 = v32 + 6;
    v7 = v32;
  }
  v11 = v7[7];
  if ( !v11 )
    goto LABEL_26;
LABEL_9:
  v12 = *a1;
  v13 = (__int64)v10;
  do
  {
    while ( 1 )
    {
      v14 = *(_QWORD *)(v11 + 16);
      v15 = *(_QWORD *)(v11 + 24);
      if ( *(_DWORD *)(v11 + 32) >= v12 )
        break;
      v11 = *(_QWORD *)(v11 + 24);
      if ( !v15 )
        goto LABEL_13;
    }
    v13 = v11;
    v11 = *(_QWORD *)(v11 + 16);
  }
  while ( v14 );
LABEL_13:
  if ( (_QWORD *)v13 == v10 || v12 < *(_DWORD *)(v13 + 32) )
    goto LABEL_15;
LABEL_20:
  *(_DWORD *)(v13 + 36) = a2;
  return a2;
}
