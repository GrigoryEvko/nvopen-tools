// Function: sub_1598370
// Address: 0x1598370
//
__int64 __fastcall sub_1598370(__int64 a1, __int64 a2, _QWORD *a3)
{
  unsigned int v4; // r14d
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // rax
  __int64 v10; // r13
  int v11; // eax
  unsigned int v12; // r8d
  __int64 v13; // r10
  __int64 v14; // r9
  int v15; // r11d
  unsigned int i; // edx
  __int64 v17; // rcx
  __int64 v18; // rax
  __int64 v19; // rsi
  unsigned int v20; // edx
  __int64 v21; // rsi
  unsigned int v22; // eax
  unsigned int v23; // eax
  __int64 v24; // r12
  __int64 v25; // rsi
  __int64 v26; // rbx
  __int64 v27; // r12
  __int64 v28; // rsi
  __int64 v29; // rbx
  __int64 v30; // rdi
  char v31; // al
  __int64 v32; // rdi
  char v33; // al
  __int64 v34; // rcx
  __int64 v35; // [rsp+0h] [rbp-B0h]
  __int64 v36; // [rsp+0h] [rbp-B0h]
  __int64 v37; // [rsp+0h] [rbp-B0h]
  int v38; // [rsp+8h] [rbp-A8h]
  __int64 v39; // [rsp+8h] [rbp-A8h]
  __int64 v40; // [rsp+8h] [rbp-A8h]
  unsigned int v41; // [rsp+14h] [rbp-9Ch]
  int v42; // [rsp+14h] [rbp-9Ch]
  int v43; // [rsp+14h] [rbp-9Ch]
  __int64 v44; // [rsp+18h] [rbp-98h]
  unsigned int v45; // [rsp+18h] [rbp-98h]
  unsigned int v46; // [rsp+18h] [rbp-98h]
  unsigned int v47; // [rsp+20h] [rbp-90h]
  __int64 v48; // [rsp+20h] [rbp-90h]
  __int64 v49; // [rsp+20h] [rbp-90h]
  __int64 v50; // [rsp+28h] [rbp-88h]
  unsigned int v51; // [rsp+28h] [rbp-88h]
  unsigned int v52; // [rsp+28h] [rbp-88h]
  __int64 v53; // [rsp+30h] [rbp-80h]
  __int64 v54; // [rsp+38h] [rbp-78h]
  __int64 v55; // [rsp+48h] [rbp-68h] BYREF
  __int64 v56; // [rsp+50h] [rbp-60h]
  __int64 v57; // [rsp+68h] [rbp-48h] BYREF
  __int64 v58; // [rsp+70h] [rbp-40h]

  v4 = *(_DWORD *)(a1 + 24);
  if ( !v4 )
  {
    *a3 = 0;
    return v4;
  }
  v54 = *(_QWORD *)(a1 + 8);
  v53 = sub_16982B0(a1, a2);
  v9 = sub_16982C0(a1, a2, v7, v8);
  v10 = v9;
  if ( v53 == v9 )
  {
    sub_169C630(&v55, v9, 1);
    sub_169C630(&v57, v10, 2);
  }
  else
  {
    sub_1699170(&v55, v53, 1);
    sub_1699170(&v57, v53, 2);
  }
  v11 = sub_16A4DD0(a2);
  v12 = v4 - 1;
  v13 = 0;
  v14 = a2 + 8;
  v15 = 1;
  for ( i = (v4 - 1) & v11; ; i = v12 & v20 )
  {
    v17 = v54 + 40LL * i;
    v18 = *(_QWORD *)(a2 + 8);
    v19 = *(_QWORD *)(v17 + 8);
    if ( v18 == v19 )
    {
      v35 = v54 + 40LL * i;
      v21 = v17 + 8;
      v38 = v15;
      v41 = i;
      v44 = v13;
      v47 = v12;
      v50 = v14;
      if ( v10 == v18 )
      {
        v23 = sub_169CB90(v14, v21);
        v17 = v35;
        v15 = v38;
        i = v41;
        v13 = v44;
        v4 = v23;
        v12 = v47;
        v14 = v50;
      }
      else
      {
        v22 = sub_1698510(v14, v21);
        v14 = v50;
        v12 = v47;
        v13 = v44;
        i = v41;
        v4 = v22;
        v15 = v38;
        v17 = v35;
      }
      if ( (_BYTE)v4 )
      {
        *a3 = v17;
        goto LABEL_15;
      }
      v19 = *(_QWORD *)(v17 + 8);
    }
    if ( v55 == v19 )
      break;
LABEL_9:
    if ( v19 == v57 )
    {
      v37 = v14;
      v32 = v17 + 8;
      v40 = v17;
      v43 = v15;
      v46 = i;
      v49 = v13;
      v52 = v12;
      if ( v19 == v10 )
      {
        v33 = sub_169CB90(v32, &v57);
        v14 = v37;
        v34 = v40;
        v15 = v43;
        i = v46;
        v13 = v49;
        v12 = v52;
      }
      else
      {
        v33 = sub_1698510(v32, &v57);
        v12 = v52;
        v13 = v49;
        i = v46;
        v15 = v43;
        v34 = v40;
        v14 = v37;
      }
      if ( !v13 && v33 )
        v13 = v34;
    }
    v20 = v15 + i;
    ++v15;
  }
  v45 = i;
  v30 = v17 + 8;
  v36 = v14;
  v39 = v17;
  v42 = v15;
  v48 = v13;
  v51 = v12;
  if ( v10 == v19 )
  {
    v31 = sub_169CB90(v30, &v55);
    v14 = v36;
    v17 = v39;
    v15 = v42;
    i = v45;
    v13 = v48;
    v12 = v51;
  }
  else
  {
    v31 = sub_1698510(v30, &v55);
    v12 = v51;
    v13 = v48;
    i = v45;
    v15 = v42;
    v17 = v39;
    v14 = v36;
  }
  if ( !v31 )
  {
    v19 = *(_QWORD *)(v17 + 8);
    goto LABEL_9;
  }
  if ( !v13 )
    v13 = v17;
  v4 = 0;
  *a3 = v13;
LABEL_15:
  if ( v10 == v57 )
  {
    v27 = v58;
    if ( v58 )
    {
      v28 = 32LL * *(_QWORD *)(v58 - 8);
      v29 = v58 + v28;
      if ( v58 != v58 + v28 )
      {
        do
        {
          v29 -= 32;
          sub_127D120((_QWORD *)(v29 + 8));
        }
        while ( v27 != v29 );
      }
      j_j_j___libc_free_0_0(v27 - 8);
    }
  }
  else
  {
    sub_1698460(&v57);
  }
  if ( v10 == v55 )
  {
    v24 = v56;
    if ( v56 )
    {
      v25 = 32LL * *(_QWORD *)(v56 - 8);
      v26 = v56 + v25;
      if ( v56 != v56 + v25 )
      {
        do
        {
          v26 -= 32;
          sub_127D120((_QWORD *)(v26 + 8));
        }
        while ( v24 != v26 );
      }
      j_j_j___libc_free_0_0(v24 - 8);
    }
  }
  else
  {
    sub_1698460(&v55);
  }
  return v4;
}
