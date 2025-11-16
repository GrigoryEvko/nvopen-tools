// Function: sub_AC6AC0
// Address: 0xac6ac0
//
__int64 __fastcall sub_AC6AC0(__int64 a1, int *a2, __int64 *a3)
{
  int v3; // r13d
  __int64 v6; // r14
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r12
  unsigned int v11; // eax
  int v12; // esi
  int v13; // r10d
  _QWORD *v14; // r8
  __int64 v15; // rcx
  int v16; // r13d
  _DWORD *v17; // r9
  unsigned int i; // r14d
  __int64 v19; // rdx
  int v20; // eax
  unsigned int v21; // r14d
  __int64 v22; // rsi
  unsigned __int8 v23; // al
  char v24; // al
  __int64 v25; // rax
  __int64 v26; // rdi
  char v27; // al
  _DWORD *v28; // rdx
  _QWORD *v29; // [rsp+8h] [rbp-C8h]
  __int64 v30; // [rsp+8h] [rbp-C8h]
  __int64 v31; // [rsp+8h] [rbp-C8h]
  int v32; // [rsp+10h] [rbp-C0h]
  __int64 v33; // [rsp+10h] [rbp-C0h]
  int v34; // [rsp+18h] [rbp-B8h]
  _QWORD *v35; // [rsp+18h] [rbp-B8h]
  int v36; // [rsp+18h] [rbp-B8h]
  _DWORD *v37; // [rsp+20h] [rbp-B0h]
  _DWORD *v38; // [rsp+20h] [rbp-B0h]
  _DWORD *v39; // [rsp+20h] [rbp-B0h]
  __int64 v40; // [rsp+28h] [rbp-A8h]
  _DWORD *v41; // [rsp+28h] [rbp-A8h]
  _QWORD *v42; // [rsp+28h] [rbp-A8h]
  __int64 v43; // [rsp+30h] [rbp-A0h]
  unsigned __int8 v45; // [rsp+38h] [rbp-98h]
  _QWORD v46[4]; // [rsp+40h] [rbp-90h] BYREF
  int v47; // [rsp+60h] [rbp-70h]
  char v48; // [rsp+64h] [rbp-6Ch]
  _QWORD v49[3]; // [rsp+68h] [rbp-68h] BYREF
  __int64 v50; // [rsp+80h] [rbp-50h] BYREF
  _QWORD v51[9]; // [rsp+88h] [rbp-48h] BYREF

  v3 = *(_DWORD *)(a1 + 24);
  if ( !v3 )
  {
    *a3 = 0;
    return 0;
  }
  v43 = *(_QWORD *)(a1 + 8);
  v6 = sub_C33690();
  v10 = sub_C33340(a1, a2, v7, v8, v9);
  if ( v6 == v10 )
    sub_C3C5A0(&v50, v6, 1);
  else
    sub_C36740(&v50, v6, 1);
  v48 = 1;
  v47 = -1;
  if ( v50 == v10 )
    sub_C3C840(v49, &v50);
  else
    sub_C338E0(v49, &v50);
  sub_91D830(&v50);
  if ( v6 == v10 )
    sub_C3C5A0(v46, v10, 2);
  else
    sub_C36740(v46, v6, 2);
  BYTE4(v50) = 0;
  LODWORD(v50) = -2;
  if ( v10 == v46[0] )
    sub_C3C840(v51, v46);
  else
    sub_C338E0(v51, v46);
  sub_91D830(v46);
  v11 = sub_C42050(a2 + 2);
  v12 = *a2;
  v13 = 1;
  v14 = v51;
  v15 = (__int64)(a2 + 2);
  v16 = v3 - 1;
  v17 = 0;
  for ( i = v16
          & (((0xBF58476D1CE4E5B9LL
             * (v11 | ((unsigned __int64)((unsigned int)(*((_BYTE *)a2 + 4) == 0) + 37 * *a2 - 1) << 32))) >> 31)
           ^ (484763065 * v11)); ; i = v16 & v21 )
  {
    v19 = v43 + 40LL * i;
    v20 = *(_DWORD *)v19;
    if ( *(_DWORD *)v19 == v12 && *((_BYTE *)a2 + 4) == *(_BYTE *)(v19 + 4) )
    {
      v22 = *((_QWORD *)a2 + 1);
      if ( v22 == *(_QWORD *)(v19 + 8) )
      {
        v29 = v14;
        v34 = v13;
        v37 = v17;
        v40 = v15;
        if ( v10 == v22 )
        {
          v23 = sub_C3E590(v15);
          v14 = v29;
          v19 = v43 + 40LL * i;
          v13 = v34;
          v17 = v37;
          v15 = v40;
        }
        else
        {
          v23 = sub_C33D00(v15);
          v15 = v40;
          v17 = v37;
          v13 = v34;
          v19 = v43 + 40LL * i;
          v14 = v29;
        }
        if ( v23 )
        {
          *a3 = v19;
          goto LABEL_22;
        }
        v20 = *(_DWORD *)v19;
      }
    }
    if ( v47 == v20 && *(_BYTE *)(v19 + 4) == v48 )
      break;
LABEL_14:
    if ( (_DWORD)v50 == v20 && *(_BYTE *)(v19 + 4) == BYTE4(v50) )
    {
      v25 = *(_QWORD *)(v19 + 8);
      if ( v25 == v51[0] )
      {
        v31 = v15;
        v26 = v19 + 8;
        v33 = v19;
        v36 = v13;
        v39 = v17;
        v42 = v14;
        if ( v10 == v25 )
        {
          v27 = sub_C3E590(v26);
          v15 = v31;
          v28 = (_DWORD *)v33;
          v13 = v36;
          v17 = v39;
          v14 = v42;
        }
        else
        {
          v27 = sub_C33D00(v26);
          v14 = v42;
          v17 = v39;
          v13 = v36;
          v28 = (_DWORD *)v33;
          v15 = v31;
        }
        if ( v27 )
        {
          if ( !v17 )
            v17 = v28;
        }
      }
    }
    v21 = v13 + i;
    v12 = *a2;
    ++v13;
  }
  v35 = v14;
  v38 = v17;
  v41 = (_DWORD *)v19;
  v30 = v15;
  v32 = v13;
  v24 = sub_AC2B80((__int64 *)(v19 + 8), v49, v19, v15, (__int64)v14);
  v19 = (__int64)v41;
  v17 = v38;
  v14 = v35;
  if ( !v24 )
  {
    v20 = *v41;
    v13 = v32;
    v15 = v30;
    goto LABEL_14;
  }
  if ( !v38 )
    v17 = v41;
  *a3 = (__int64)v17;
  v23 = 0;
LABEL_22:
  v45 = v23;
  sub_91D830(v14);
  sub_91D830(v49);
  return v45;
}
