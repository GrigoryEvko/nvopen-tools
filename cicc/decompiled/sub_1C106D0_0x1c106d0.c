// Function: sub_1C106D0
// Address: 0x1c106d0
//
__int64 __fastcall sub_1C106D0(__int64 a1, __int64 a2, __int64 a3, _BYTE *a4, __int64 a5)
{
  unsigned int v5; // r15d
  unsigned __int64 v7; // rax
  unsigned __int64 v8; // r13
  int v9; // r8d
  int v10; // r9d
  __int64 v11; // rax
  int v12; // eax
  __int64 v13; // rsi
  __int64 v14; // rax
  int v15; // eax
  __int64 *v16; // rdi
  __int64 *v17; // r12
  __int64 v19; // rax
  size_t v20; // rdi
  __int64 v21; // r12
  unsigned int v22; // eax
  __int64 v23; // rax
  __int64 *v24; // rbx
  __int64 v25; // r8
  __int64 v26; // rax
  unsigned int v27; // esi
  int v28; // r9d
  __int64 v29; // rdi
  unsigned int v30; // edx
  __int64 v31; // r13
  __int64 v32; // rcx
  __int64 v33; // rax
  __int64 *v34; // rax
  int v35; // r11d
  __int64 v36; // r10
  int v37; // ecx
  int v38; // ecx
  unsigned int v41; // [rsp+18h] [rbp-F8h]
  __int64 v43; // [rsp+28h] [rbp-E8h]
  __int64 v44; // [rsp+28h] [rbp-E8h]
  unsigned int v45; // [rsp+3Ch] [rbp-D4h] BYREF
  __int64 v46; // [rsp+40h] [rbp-D0h] BYREF
  unsigned __int64 v47; // [rsp+48h] [rbp-C8h] BYREF
  __int64 v48; // [rsp+50h] [rbp-C0h] BYREF
  __int64 v49; // [rsp+58h] [rbp-B8h]
  __int64 v50; // [rsp+60h] [rbp-B0h]
  __int64 v51; // [rsp+68h] [rbp-A8h]
  __int64 v52; // [rsp+70h] [rbp-A0h] BYREF
  __int64 v53; // [rsp+78h] [rbp-98h]
  __int64 v54; // [rsp+80h] [rbp-90h]
  __int64 v55; // [rsp+88h] [rbp-88h]
  _QWORD *v56; // [rsp+90h] [rbp-80h] BYREF
  __int64 *v57; // [rsp+98h] [rbp-78h]
  __int64 v58; // [rsp+A0h] [rbp-70h]
  __int64 v59; // [rsp+A8h] [rbp-68h]
  _BYTE *v60; // [rsp+B0h] [rbp-60h] BYREF
  __int64 v61; // [rsp+B8h] [rbp-58h]
  _BYTE v62[80]; // [rsp+C0h] [rbp-50h] BYREF

  v5 = 0;
  v7 = sub_157EBA0(a2);
  if ( !v7 )
    return v5;
  v8 = v7;
  v52 = 0;
  v60 = v62;
  v61 = 0x400000000LL;
  v53 = 0;
  v54 = 0;
  v55 = 0;
  v47 = v7;
  v48 = 1;
  v49 = 0;
  v50 = 0;
  v51 = 0;
  sub_1353F00((__int64)&v48, 0);
  sub_1A97120((__int64)&v48, (__int64 *)&v47, &v56);
  LODWORD(v50) = v50 + 1;
  if ( *v56 != -8 )
    --HIDWORD(v50);
  *v56 = v47;
  v11 = (unsigned int)v61;
  if ( (unsigned int)v61 >= HIDWORD(v61) )
  {
    sub_16CD150((__int64)&v60, v62, 0, 8, v9, v10);
    v11 = (unsigned int)v61;
  }
  *(_QWORD *)&v60[8 * v11] = v8;
  v45 = 0;
  v12 = v61 + 1;
  LODWORD(v61) = v61 + 1;
  while ( 1 )
  {
    if ( !v12 )
    {
      v56 = (_QWORD *)a2;
      sub_1C0A020(a5 + 40, (__int64 *)&v56);
      goto LABEL_23;
    }
    v13 = *(_QWORD *)&v60[8 * v12 - 8];
    LODWORD(v61) = v12 - 1;
    if ( *(_BYTE *)(v13 + 16) == 54 )
    {
      v14 = **(_QWORD **)(v13 - 24);
      if ( *(_BYTE *)(v14 + 8) == 16 )
        v14 = **(_QWORD **)(v14 + 16);
      v15 = *(_DWORD *)(v14 + 8) >> 8;
      if ( v15 == 5 || !v15 )
        *a4 = 1;
    }
    sub_1C0CC70(*(_QWORD **)(a1 + 8), v13, &v45, (__int64)&v48, (__int64)&v60, (__int64)&v52);
    if ( v45 )
      break;
    v12 = v61;
  }
  v41 = v45;
  v56 = (_QWORD *)a2;
  v23 = sub_1C0A020(a5 + 40, (__int64 *)&v56)[1];
  if ( *(_DWORD *)(v23 + 16) )
  {
LABEL_23:
    v19 = sub_1C0A150(a5, a2);
    v20 = *(_QWORD *)(a1 + 8);
    v56 = 0;
    v21 = v19;
    v57 = 0;
    v58 = 0;
    v59 = 0;
    v22 = sub_1C0EF50(v20, v45, (__int64)&v52, a2, (__int64)&v56);
    v45 = v22;
    if ( v22 )
    {
      v16 = v57;
      v5 = 1;
      if ( !*(_DWORD *)(v21 + 16) )
        *(_DWORD *)(v21 + 16) = v22;
      goto LABEL_18;
    }
    v16 = v57;
    v17 = &v57[(unsigned int)v59];
    if ( !(_DWORD)v58 || v57 == v17 )
      goto LABEL_17;
    v24 = v57;
    while ( *v24 == -16 || *v24 == -8 )
    {
      if ( ++v24 == v17 )
        goto LABEL_17;
    }
    if ( v24 == v17 )
    {
LABEL_17:
      v5 = 0;
LABEL_18:
      j___libc_free_0(v16);
      goto LABEL_19;
    }
    v25 = a2;
LABEL_35:
    v26 = *v24;
    v27 = *(_DWORD *)(a3 + 24);
    v46 = *v24;
    if ( v27 )
    {
      v28 = v27 - 1;
      v29 = *(_QWORD *)(a3 + 8);
      v30 = (v27 - 1) & (((unsigned int)v26 >> 9) ^ ((unsigned int)v26 >> 4));
      v31 = v29 + 40LL * v30;
      v32 = *(_QWORD *)v31;
      if ( v26 == *(_QWORD *)v31 )
      {
LABEL_37:
        v33 = *(unsigned int *)(v31 + 16);
        if ( (unsigned int)v33 >= *(_DWORD *)(v31 + 20) )
        {
          v43 = v25;
          sub_16CD150(v31 + 8, (const void *)(v31 + 24), 0, 8, v25, v28);
          v25 = v43;
          v34 = (__int64 *)(*(_QWORD *)(v31 + 8) + 8LL * *(unsigned int *)(v31 + 16));
        }
        else
        {
          v34 = (__int64 *)(*(_QWORD *)(v31 + 8) + 8 * v33);
        }
        goto LABEL_39;
      }
      v35 = 1;
      v36 = 0;
      while ( v32 != -8 )
      {
        if ( !v36 && v32 == -16 )
          v36 = v31;
        v30 = v28 & (v35 + v30);
        v31 = v29 + 40LL * v30;
        v32 = *(_QWORD *)v31;
        if ( v26 == *(_QWORD *)v31 )
          goto LABEL_37;
        ++v35;
      }
      v37 = *(_DWORD *)(a3 + 16);
      if ( v36 )
        v31 = v36;
      ++*(_QWORD *)a3;
      v38 = v37 + 1;
      if ( 4 * v38 < 3 * v27 )
      {
        if ( v27 - *(_DWORD *)(a3 + 20) - v38 > v27 >> 3 )
        {
LABEL_52:
          *(_DWORD *)(a3 + 16) = v38;
          if ( *(_QWORD *)v31 != -8 )
            --*(_DWORD *)(a3 + 20);
          *(_QWORD *)v31 = v26;
          v34 = (__int64 *)(v31 + 24);
          *(_QWORD *)(v31 + 8) = v31 + 24;
          *(_QWORD *)(v31 + 16) = 0x200000000LL;
LABEL_39:
          *v34 = v25;
          ++*(_DWORD *)(v31 + 16);
          while ( ++v24 != v17 )
          {
            if ( *v24 != -16 && *v24 != -8 )
            {
              if ( v24 != v17 )
                goto LABEL_35;
              break;
            }
          }
          v16 = v57;
          goto LABEL_17;
        }
        v44 = v25;
LABEL_57:
        sub_1C0EAB0(a3, v27);
        sub_1C09750(a3, &v46, &v47);
        v31 = v47;
        v26 = v46;
        v25 = v44;
        v38 = *(_DWORD *)(a3 + 16) + 1;
        goto LABEL_52;
      }
    }
    else
    {
      ++*(_QWORD *)a3;
    }
    v44 = v25;
    v27 *= 2;
    goto LABEL_57;
  }
  *(_DWORD *)(v23 + 16) = v41;
  v5 = 1;
LABEL_19:
  j___libc_free_0(v53);
  j___libc_free_0(v49);
  if ( v60 != v62 )
    _libc_free((unsigned __int64)v60);
  return v5;
}
