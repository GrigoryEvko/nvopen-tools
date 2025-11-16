// Function: sub_3154710
// Address: 0x3154710
//
_QWORD *__fastcall sub_3154710(_QWORD *a1, __int64 *a2)
{
  _QWORD *v2; // r15
  _QWORD *v4; // r12
  unsigned __int64 v5; // rcx
  unsigned __int64 v6; // rdx
  _QWORD *v7; // rax
  _BOOL4 v8; // r8d
  __int64 v9; // rax
  __int64 v10; // r9
  __int64 *v11; // rdx
  char v12; // r8
  __int64 v13; // r13
  unsigned __int64 v14; // rax
  __int64 v15; // rcx
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // rax
  __int64 v19; // rdx
  int v20; // esi
  __int64 v21; // rax
  __int64 v23; // rax
  _BOOL4 v24; // [rsp+Ch] [rbp-34h]

  v2 = a1 + 1;
  v4 = (_QWORD *)a1[2];
  if ( !v4 )
  {
    v4 = a1 + 1;
LABEL_21:
    if ( v4 == (_QWORD *)a1[3] )
      goto LABEL_9;
    v23 = sub_220EF80((__int64)v4);
    if ( *(_QWORD *)(v23 + 32) < (unsigned __int64)*a2 )
      goto LABEL_9;
    return (_QWORD *)v23;
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
  v24 = v8;
  v9 = sub_22077B0(0x100u);
  v11 = (__int64 *)a2[2];
  v12 = v24;
  v13 = v9;
  v14 = *a2;
  *(_QWORD *)(v13 + 48) = v11;
  v15 = v13 + 40;
  *(_QWORD *)(v13 + 32) = v14;
  v16 = a2[1];
  *(_QWORD *)(v13 + 40) = v16;
  if ( v11 )
  {
    *v11 = v15;
    v16 = a2[1];
  }
  if ( v16 )
    *(_QWORD *)(v16 + 8) = v15;
  v17 = a2[3];
  a2[2] = 0;
  a2[1] = 0;
  *(_QWORD *)(v13 + 56) = v17;
  *(_QWORD *)(v13 + 64) = v13 + 80;
  *(_QWORD *)(v13 + 72) = 0x1000000000LL;
  if ( *((_DWORD *)a2 + 10) )
  {
    sub_3153680(v13 + 64, (char **)a2 + 4, (__int64)v11, v15, v24, v10);
    v12 = v24;
  }
  v18 = a2[24];
  v19 = v13 + 216;
  if ( v18 )
  {
    v20 = *((_DWORD *)a2 + 46);
    *(_QWORD *)(v13 + 224) = v18;
    *(_DWORD *)(v13 + 216) = v20;
    *(_QWORD *)(v13 + 232) = a2[25];
    *(_QWORD *)(v13 + 240) = a2[26];
    *(_QWORD *)(v18 + 8) = v19;
    v21 = a2[27];
    a2[24] = 0;
    *(_QWORD *)(v13 + 248) = v21;
    a2[25] = (__int64)(a2 + 23);
    a2[26] = (__int64)(a2 + 23);
    a2[27] = 0;
  }
  else
  {
    *(_DWORD *)(v13 + 216) = 0;
    *(_QWORD *)(v13 + 224) = 0;
    *(_QWORD *)(v13 + 232) = v19;
    *(_QWORD *)(v13 + 240) = v19;
    *(_QWORD *)(v13 + 248) = 0;
  }
  sub_220F040(v12, v13, v4, v2);
  ++a1[5];
  return (_QWORD *)v13;
}
