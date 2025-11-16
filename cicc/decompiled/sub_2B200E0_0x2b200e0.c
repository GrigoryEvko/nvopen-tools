// Function: sub_2B200E0
// Address: 0x2b200e0
//
bool __fastcall sub_2B200E0(_QWORD *a1, __int64 a2)
{
  __int64 v2; // r9
  char v3; // r8
  __int64 v4; // r10
  int v5; // r11d
  unsigned int v6; // edx
  __int64 *v7; // rcx
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 *v10; // r8
  bool result; // al
  __int64 v12; // r11
  __int64 v13; // rbx
  __int64 *v14; // r12
  __int64 v15; // rax
  _QWORD *v16; // r9
  __int64 *v17; // r10
  _QWORD *v18; // rsi
  __int64 v19; // r8
  _QWORD *v20; // r9
  __int64 *v21; // r10
  __int64 v22; // r8
  _QWORD *v23; // r9
  __int64 *v24; // r10
  __int64 v25; // r8
  __int64 *v26; // r11
  __int64 *v27; // r8
  _QWORD *v28; // r9
  __int64 *v29; // r10
  __int64 v30; // rdx
  __int64 v31; // rcx
  int v32; // ecx
  _QWORD *v33; // r9
  __int64 *v34; // r10
  _QWORD *v35; // rsi
  __int64 *v36; // r8
  _QWORD *v37; // r9
  __int64 v38; // rsi
  _QWORD *v39; // rdi
  _QWORD *v40; // r9
  __int64 *v41; // r10
  __int64 v42; // r11
  int v43; // ebx
  __int64 v44[3]; // [rsp+8h] [rbp-18h] BYREF

  v2 = *a1;
  v3 = *(_BYTE *)(*a1 + 88LL) & 1;
  if ( v3 )
  {
    v4 = v2 + 96;
    v5 = 3;
  }
  else
  {
    v30 = *(unsigned int *)(v2 + 104);
    v4 = *(_QWORD *)(v2 + 96);
    if ( !(_DWORD)v30 )
      goto LABEL_25;
    v5 = v30 - 1;
  }
  v6 = v5 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v7 = (__int64 *)(v4 + 72LL * v6);
  v8 = *v7;
  if ( a2 == *v7 )
    goto LABEL_4;
  v32 = 1;
  while ( v8 != -4096 )
  {
    v43 = v32 + 1;
    v6 = v5 & (v32 + v6);
    v7 = (__int64 *)(v4 + 72LL * v6);
    v8 = *v7;
    if ( a2 == *v7 )
      goto LABEL_4;
    v32 = v43;
  }
  if ( v3 )
  {
    v31 = 288;
    goto LABEL_26;
  }
  v30 = *(unsigned int *)(v2 + 104);
LABEL_25:
  v31 = 72 * v30;
LABEL_26:
  v7 = (__int64 *)(v4 + v31);
LABEL_4:
  v9 = 288;
  if ( !v3 )
    v9 = 72LL * *(unsigned int *)(v2 + 104);
  if ( v7 == (__int64 *)(v4 + v9) )
    return 0;
  v10 = (__int64 *)v7[1];
  result = 0;
  if ( !*((_DWORD *)v7 + 4) )
    return result;
  v12 = 8LL * *((unsigned int *)v7 + 4);
  v13 = a1[1];
  v14 = &v10[(unsigned __int64)v12 / 8];
  v15 = v12 >> 3;
  if ( v12 >> 5 )
  {
    v16 = *(_QWORD **)v13;
    v17 = v44;
    v18 = (_QWORD *)(*(_QWORD *)v13 + 8LL * *(_QWORD *)(v13 + 8));
    while ( 1 )
    {
      v44[0] = *v10;
      if ( v18 != sub_2B0BA00(v16, (__int64)v18, v17) )
        return v27 != v14;
      v44[0] = v27[1];
      if ( v18 != sub_2B0BA00(v28, (__int64)v18, v29) )
        return v14 != (__int64 *)(v19 + 8);
      v44[0] = *(_QWORD *)(v19 + 16);
      if ( v18 != sub_2B0BA00(v20, (__int64)v18, v21) )
        return v14 != (__int64 *)(v22 + 16);
      v44[0] = *(_QWORD *)(v22 + 24);
      if ( v18 != sub_2B0BA00(v23, (__int64)v18, v24) )
        return v14 != (__int64 *)(v25 + 24);
      v10 = (__int64 *)(v25 + 32);
      if ( v26 == v10 )
      {
        v15 = v14 - v10;
        break;
      }
    }
  }
  switch ( v15 )
  {
    case 2LL:
      v40 = *(_QWORD **)v13;
      v42 = *(_QWORD *)(v13 + 8);
      v41 = v44;
      break;
    case 3LL:
      v38 = *(_QWORD *)(v13 + 8);
      v39 = *(_QWORD **)v13;
      v44[0] = *v10;
      if ( sub_2B0D750(v39, v38, v44) )
        return v27 != v14;
      v10 = v27 + 1;
      break;
    case 1LL:
      v33 = *(_QWORD **)v13;
      v34 = v44;
      v35 = (_QWORD *)(*(_QWORD *)v13 + 8LL * *(_QWORD *)(v13 + 8));
      goto LABEL_35;
    default:
      return 0;
  }
  v35 = &v40[v42];
  v44[0] = *v10;
  if ( v35 != sub_2B0BA00(v40, (__int64)v35, v41) )
    return v14 != v36;
  v10 = v36 + 1;
LABEL_35:
  v44[0] = *v10;
  v37 = sub_2B0BA00(v33, (__int64)v35, v34);
  result = 0;
  if ( v35 != v37 )
    return v14 != v36;
  return result;
}
