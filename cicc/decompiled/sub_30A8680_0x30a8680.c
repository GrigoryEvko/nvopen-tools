// Function: sub_30A8680
// Address: 0x30a8680
//
_QWORD *__fastcall sub_30A8680(_QWORD *a1, __int64 *a2)
{
  _QWORD *v2; // r15
  _QWORD *v4; // r12
  unsigned __int64 v5; // rcx
  unsigned __int64 v6; // rdx
  _QWORD *v7; // rax
  _BOOL4 v8; // r8d
  _QWORD *v9; // rax
  _BYTE *v10; // rsi
  __int64 v11; // rbx
  __int64 v12; // rdx
  __int64 v13; // r9
  __int64 *v14; // rdx
  __int64 v15; // rax
  __int64 v16; // rcx
  char v17; // r8
  __int64 v18; // rax
  __int64 v19; // rax
  __int64 v20; // rdx
  int v21; // esi
  __int64 v22; // rax
  __int64 v24; // rax
  _BOOL4 v25; // [rsp+Ch] [rbp-34h]

  v2 = a1 + 1;
  v4 = (_QWORD *)a1[2];
  if ( !v4 )
  {
    v4 = a1 + 1;
LABEL_21:
    if ( v4 == (_QWORD *)a1[3] )
      goto LABEL_9;
    v24 = sub_220EF80((__int64)v4);
    if ( *(_QWORD *)(v24 + 32) < (unsigned __int64)*a2 )
      goto LABEL_9;
    return (_QWORD *)v24;
  }
  v5 = *a2;
  while ( 1 )
  {
    v6 = v4[4];
    v7 = (_QWORD *)v4[3];
    if ( v5 < v6 )
      v7 = (_QWORD *)v4[2];
    if ( !v7 )
      break;
    v4 = v7;
  }
  if ( v5 < v6 )
    goto LABEL_21;
  if ( v5 <= v6 )
    return v4;
LABEL_9:
  v8 = 1;
  if ( v2 != v4 )
    v8 = (unsigned __int64)*a2 < v4[4];
  v25 = v8;
  v9 = (_QWORD *)sub_22077B0(0x128u);
  v10 = (_BYTE *)a2[2];
  v11 = (__int64)v9;
  v12 = (__int64)&v10[a2[3]];
  v9[4] = *a2;
  v9[5] = a2[1];
  v9[6] = v9 + 8;
  sub_30A6F00(v9 + 6, v10, v12);
  v14 = (__int64 *)a2[7];
  v15 = a2[6];
  v16 = v11 + 80;
  v17 = v25;
  *(_QWORD *)(v11 + 88) = v14;
  *(_QWORD *)(v11 + 80) = v15;
  if ( v14 )
  {
    *v14 = v16;
    v15 = a2[6];
  }
  if ( v15 )
    *(_QWORD *)(v15 + 8) = v16;
  v18 = a2[8];
  a2[7] = 0;
  a2[6] = 0;
  *(_QWORD *)(v11 + 96) = v18;
  *(_QWORD *)(v11 + 104) = v11 + 120;
  *(_QWORD *)(v11 + 112) = 0x1000000000LL;
  if ( *((_DWORD *)a2 + 20) )
  {
    sub_30A6A60(v11 + 104, (char **)a2 + 9, (__int64)v14, v16, v25, v13);
    v17 = v25;
  }
  v19 = a2[29];
  v20 = v11 + 256;
  if ( v19 )
  {
    v21 = *((_DWORD *)a2 + 56);
    *(_QWORD *)(v11 + 264) = v19;
    *(_DWORD *)(v11 + 256) = v21;
    *(_QWORD *)(v11 + 272) = a2[30];
    *(_QWORD *)(v11 + 280) = a2[31];
    *(_QWORD *)(v19 + 8) = v20;
    v22 = a2[32];
    a2[29] = 0;
    *(_QWORD *)(v11 + 288) = v22;
    a2[30] = (__int64)(a2 + 28);
    a2[31] = (__int64)(a2 + 28);
    a2[32] = 0;
  }
  else
  {
    *(_DWORD *)(v11 + 256) = 0;
    *(_QWORD *)(v11 + 264) = 0;
    *(_QWORD *)(v11 + 272) = v20;
    *(_QWORD *)(v11 + 280) = v20;
    *(_QWORD *)(v11 + 288) = 0;
  }
  sub_220F040(v17, v11, v4, v2);
  ++a1[5];
  return (_QWORD *)v11;
}
