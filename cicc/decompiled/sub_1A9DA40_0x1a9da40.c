// Function: sub_1A9DA40
// Address: 0x1a9da40
//
__int64 __fastcall sub_1A9DA40(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rax
  __int64 v6; // rdi
  __int64 *v7; // rsi
  __int64 v8; // rax
  void *v9; // rdi
  __int64 v10; // r12
  __int64 v11; // rdx
  __int64 v12; // rax
  unsigned __int64 v13; // r14
  char *v14; // rcx
  _BYTE *v15; // rax
  _BYTE *v16; // rsi
  signed __int64 v17; // r12
  __int64 v18; // rsi
  char *v19; // rax
  char *v20; // rsi
  __int64 *v21; // r12
  __int64 *v22; // r15
  __int64 v23; // r8
  __int64 *v24; // r10
  int v25; // r11d
  unsigned int v26; // eax
  __int64 *v27; // rdi
  __int64 v28; // rcx
  unsigned int v29; // esi
  int v30; // edx
  int v31; // eax
  __int64 v32; // rax
  _BYTE *v33; // rsi
  __int64 v35; // [rsp+18h] [rbp-88h] BYREF
  __int64 v36; // [rsp+20h] [rbp-80h] BYREF
  __int64 *v37; // [rsp+28h] [rbp-78h] BYREF
  __int64 v38; // [rsp+30h] [rbp-70h] BYREF
  void *v39; // [rsp+38h] [rbp-68h]
  __int64 v40; // [rsp+40h] [rbp-60h]
  __int64 v41; // [rsp+48h] [rbp-58h]
  char *v42; // [rsp+50h] [rbp-50h]
  char *v43; // [rsp+58h] [rbp-48h]
  char *v44; // [rsp+60h] [rbp-40h]

  v5 = *(_QWORD *)(a1 + 40);
  v6 = a2 + 168;
  v7 = &v35;
  v35 = v5;
  v8 = sub_1A9D7E0(v6, &v35);
  v9 = 0;
  v38 = 0;
  v10 = v8;
  v39 = 0;
  v40 = 0;
  v41 = 0;
  j___libc_free_0(0);
  v12 = *(unsigned int *)(v10 + 24);
  LODWORD(v41) = v12;
  if ( (_DWORD)v12 )
  {
    v39 = (void *)sub_22077B0(8 * v12);
    v9 = v39;
    v40 = *(_QWORD *)(v10 + 16);
    v7 = *(__int64 **)(v10 + 8);
    memcpy(v39, v7, 8LL * (unsigned int)v41);
  }
  else
  {
    v39 = 0;
    v40 = 0;
  }
  v13 = *(_QWORD *)(v10 + 40) - *(_QWORD *)(v10 + 32);
  v42 = 0;
  v43 = 0;
  v44 = 0;
  if ( v13 )
  {
    if ( v13 > 0x7FFFFFFFFFFFFFF8LL )
      sub_4261EA(v9, v7, v11);
    v14 = (char *)sub_22077B0(v13);
  }
  else
  {
    v14 = 0;
  }
  v43 = v14;
  v44 = &v14[v13];
  v15 = *(_BYTE **)(v10 + 40);
  v42 = v14;
  v16 = *(_BYTE **)(v10 + 32);
  v17 = v15 - v16;
  if ( v15 != v16 )
    v14 = (char *)memmove(v14, v16, v15 - v16);
  v18 = *(_QWORD *)(a1 + 24);
  v43 = &v14[v17];
  sub_1A97C40(
    (_QWORD *)(*(_QWORD *)(v35 + 40) & 0xFFFFFFFFFFFFFFF8LL),
    (_QWORD *)(v18 & 0xFFFFFFFFFFFFFFF8LL),
    (__int64)&v38);
  v36 = a1;
  if ( (unsigned __int8)sub_1A97120((__int64)&v38, &v36, &v37) )
  {
    *v37 = -16;
    LODWORD(v40) = v40 - 1;
    ++HIDWORD(v40);
    v19 = (char *)sub_1A94EF0(v42, (__int64)v43, &v36);
    v20 = v19 + 8;
    if ( v43 != v19 + 8 )
    {
      memmove(v19, v20, v43 - v20);
      v20 = v43;
    }
    v21 = (__int64 *)v42;
    v22 = (__int64 *)(v20 - 8);
    v43 = v20 - 8;
    if ( v20 - 8 == v42 )
    {
LABEL_38:
      j_j___libc_free_0(v22, v44 - (char *)v22);
      return j___libc_free_0(v39);
    }
LABEL_17:
    while ( 1 )
    {
      v29 = *(_DWORD *)(a3 + 24);
      if ( !v29 )
        break;
      v23 = *(_QWORD *)(a3 + 8);
      v24 = 0;
      v25 = 1;
      v26 = (v29 - 1) & (((unsigned int)*v21 >> 9) ^ ((unsigned int)*v21 >> 4));
      v27 = (__int64 *)(v23 + 8LL * v26);
      v28 = *v27;
      if ( *v21 == *v27 )
      {
LABEL_16:
        if ( v22 == ++v21 )
          goto LABEL_36;
      }
      else
      {
        while ( v28 != -8 )
        {
          if ( v28 != -16 || v24 )
            v27 = v24;
          v26 = (v29 - 1) & (v25 + v26);
          v28 = *(_QWORD *)(v23 + 8LL * v26);
          if ( *v21 == v28 )
            goto LABEL_16;
          ++v25;
          v24 = v27;
          v27 = (__int64 *)(v23 + 8LL * v26);
        }
        v31 = *(_DWORD *)(a3 + 16);
        if ( !v24 )
          v24 = v27;
        ++*(_QWORD *)a3;
        v30 = v31 + 1;
        if ( 4 * (v31 + 1) >= 3 * v29 )
          goto LABEL_19;
        if ( v29 - *(_DWORD *)(a3 + 20) - v30 <= v29 >> 3 )
          goto LABEL_20;
LABEL_30:
        *(_DWORD *)(a3 + 16) = v30;
        if ( *v24 != -8 )
          --*(_DWORD *)(a3 + 20);
        v32 = *v21;
        *v24 = *v21;
        v33 = *(_BYTE **)(a3 + 40);
        if ( v33 == *(_BYTE **)(a3 + 48) )
        {
          sub_1287830(a3 + 32, v33, v21);
          goto LABEL_16;
        }
        if ( v33 )
        {
          *(_QWORD *)v33 = v32;
          v33 = *(_BYTE **)(a3 + 40);
        }
        ++v21;
        *(_QWORD *)(a3 + 40) = v33 + 8;
        if ( v22 == v21 )
        {
LABEL_36:
          v22 = (__int64 *)v42;
          goto LABEL_37;
        }
      }
    }
    ++*(_QWORD *)a3;
LABEL_19:
    v29 *= 2;
LABEL_20:
    sub_1353F00(a3, v29);
    sub_1A97120(a3, v21, &v37);
    v24 = v37;
    v30 = *(_DWORD *)(a3 + 16) + 1;
    goto LABEL_30;
  }
  v22 = (__int64 *)v43;
  v21 = (__int64 *)v42;
  if ( v42 != v43 )
    goto LABEL_17;
LABEL_37:
  if ( v22 )
    goto LABEL_38;
  return j___libc_free_0(v39);
}
