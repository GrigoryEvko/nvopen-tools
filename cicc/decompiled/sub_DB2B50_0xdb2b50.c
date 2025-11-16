// Function: sub_DB2B50
// Address: 0xdb2b50
//
__int64 __fastcall sub_DB2B50(__int64 a1, __int64 a2)
{
  __int64 v4; // rax
  __int64 v5; // rsi
  unsigned int v6; // ecx
  __int64 *v7; // rdx
  __int64 v8; // r10
  __int64 result; // rax
  int v10; // edx
  __int64 v11; // rdx
  bool v12; // zf
  unsigned int v13; // eax
  _QWORD *v14; // r10
  _QWORD *v15; // rsi
  __int64 v16; // rdi
  unsigned __int16 v17; // cx
  unsigned int v18; // esi
  __int64 v19; // r10
  unsigned int v20; // ebx
  unsigned int v21; // edi
  _QWORD *v22; // rcx
  __int64 v23; // r8
  int v24; // r15d
  _QWORD *v25; // rdx
  int v26; // edi
  int v27; // ecx
  __int64 v28; // rax
  __int64 v29; // rcx
  __int64 v30; // r8
  __int64 v31; // r9
  __int64 *v32; // r12
  __int64 *v33; // r15
  __int64 v34; // rbx
  __int64 *v35; // rax
  char v36; // dl
  __int64 v37; // rax
  unsigned __int64 v38; // rdx
  int v39; // r11d
  int v40; // ebx
  int v41; // ebx
  __int64 v42; // r12
  unsigned int v43; // esi
  __int64 v44; // r11
  int v45; // r10d
  _QWORD *v46; // rdi
  int v47; // r11d
  int v48; // r11d
  __int64 v49; // r12
  _QWORD *v50; // r10
  __int64 v51; // rbx
  int v52; // esi
  __int64 v53; // rdi
  unsigned __int8 v54; // [rsp+8h] [rbp-108h]
  unsigned __int8 v55; // [rsp+8h] [rbp-108h]
  _BYTE v56[2]; // [rsp+1Eh] [rbp-F2h] BYREF
  _BYTE *v57; // [rsp+20h] [rbp-F0h]
  _QWORD *v58; // [rsp+28h] [rbp-E8h] BYREF
  __int64 v59; // [rsp+30h] [rbp-E0h]
  _QWORD v60[8]; // [rsp+38h] [rbp-D8h] BYREF
  __int64 v61; // [rsp+78h] [rbp-98h] BYREF
  __int64 *v62; // [rsp+80h] [rbp-90h]
  __int64 v63; // [rsp+88h] [rbp-88h]
  int v64; // [rsp+90h] [rbp-80h]
  char v65; // [rsp+94h] [rbp-7Ch]
  __int64 v66; // [rsp+98h] [rbp-78h] BYREF

  v4 = *(unsigned int *)(a1 + 88);
  v5 = *(_QWORD *)(a1 + 72);
  if ( (_DWORD)v4 )
  {
    v6 = (v4 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v7 = (__int64 *)(v5 + 16LL * v6);
    v8 = *v7;
    if ( a2 == *v7 )
    {
LABEL_3:
      if ( v7 != (__int64 *)(v5 + 16 * v4) )
        return *((unsigned __int8 *)v7 + 8);
    }
    else
    {
      v10 = 1;
      while ( v8 != -4096 )
      {
        v39 = v10 + 1;
        v6 = (v4 - 1) & (v10 + v6);
        v7 = (__int64 *)(v5 + 16LL * v6);
        v8 = *v7;
        if ( a2 == *v7 )
          goto LABEL_3;
        v10 = v39;
      }
    }
  }
  v11 = (__int64)v56;
  v12 = *(_WORD *)(a2 + 24) == 8;
  v56[0] = 0;
  v58 = v60;
  v59 = 0x800000000LL;
  v62 = &v66;
  v57 = v56;
  v63 = 0x100000008LL;
  v64 = 0;
  v65 = 1;
  v66 = a2;
  v61 = 1;
  if ( v12 )
  {
    v56[0] = 1;
    v13 = 0;
  }
  else
  {
    v60[0] = a2;
    v13 = 1;
    LODWORD(v59) = 1;
  }
  v14 = v60;
  while ( 1 )
  {
    v15 = &v14[v13];
    if ( !v13 )
      break;
    while ( 1 )
    {
      if ( *(_BYTE *)v11 )
        goto LABEL_16;
      v16 = *(v15 - 1);
      LODWORD(v59) = --v13;
      v17 = *(_WORD *)(v16 + 24);
      if ( v17 > 0xEu )
      {
        if ( v17 != 15 )
          BUG();
        goto LABEL_15;
      }
      if ( v17 > 1u )
        break;
LABEL_15:
      --v15;
      if ( !v13 )
        goto LABEL_16;
    }
    v28 = sub_D960E0(v16);
    v32 = (__int64 *)(v28 + 8 * v11);
    v33 = (__int64 *)v28;
    if ( (__int64 *)v28 != v32 )
    {
      while ( 1 )
      {
        v34 = *v33;
        if ( !v65 )
          goto LABEL_42;
        v35 = v62;
        v29 = HIDWORD(v63);
        v11 = (__int64)&v62[HIDWORD(v63)];
        if ( v62 != (__int64 *)v11 )
        {
          while ( v34 != *v35 )
          {
            if ( (__int64 *)v11 == ++v35 )
              goto LABEL_47;
          }
          goto LABEL_39;
        }
LABEL_47:
        if ( HIDWORD(v63) < (unsigned int)v63 )
        {
          v29 = (unsigned int)++HIDWORD(v63);
          *(_QWORD *)v11 = v34;
          ++v61;
          if ( *(_WORD *)(v34 + 24) != 8 )
          {
LABEL_44:
            v37 = (unsigned int)v59;
            v29 = HIDWORD(v59);
            v38 = (unsigned int)v59 + 1LL;
            if ( v38 > HIDWORD(v59) )
            {
              sub_C8D5F0((__int64)&v58, v60, v38, 8u, v30, v31);
              v37 = (unsigned int)v59;
            }
            v58[v37] = v34;
            LODWORD(v59) = v59 + 1;
            goto LABEL_39;
          }
        }
        else
        {
LABEL_42:
          sub_C8CC70((__int64)&v61, *v33, v11, v29, v30, v31);
          if ( !v36 )
            goto LABEL_39;
          if ( *(_WORD *)(v34 + 24) != 8 )
            goto LABEL_44;
        }
        *v57 = 1;
LABEL_39:
        v11 = (__int64)v57;
        if ( !*v57 && v32 != ++v33 )
          continue;
        goto LABEL_41;
      }
    }
    v11 = (__int64)v57;
LABEL_41:
    v14 = v58;
    v13 = v59;
  }
LABEL_16:
  if ( !v65 )
  {
    _libc_free(v62, v15);
    v14 = v58;
  }
  if ( v14 != v60 )
    _libc_free(v14, v15);
  v18 = *(_DWORD *)(a1 + 88);
  result = v56[0];
  if ( !v18 )
  {
    ++*(_QWORD *)(a1 + 64);
    goto LABEL_55;
  }
  v19 = *(_QWORD *)(a1 + 72);
  v20 = ((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4);
  v21 = (v18 - 1) & v20;
  v22 = (_QWORD *)(v19 + 16LL * v21);
  v23 = *v22;
  if ( a2 == *v22 )
    return result;
  v24 = 1;
  v25 = 0;
  while ( v23 != -4096 )
  {
    if ( !v25 && v23 == -8192 )
      v25 = v22;
    v21 = (v18 - 1) & (v24 + v21);
    v22 = (_QWORD *)(v19 + 16LL * v21);
    v23 = *v22;
    if ( a2 == *v22 )
      return result;
    ++v24;
  }
  v26 = *(_DWORD *)(a1 + 80);
  if ( !v25 )
    v25 = v22;
  ++*(_QWORD *)(a1 + 64);
  v27 = v26 + 1;
  if ( 4 * (v26 + 1) >= 3 * v18 )
  {
LABEL_55:
    v54 = result;
    sub_DB2970(a1 + 64, 2 * v18);
    v40 = *(_DWORD *)(a1 + 88);
    if ( v40 )
    {
      v41 = v40 - 1;
      v42 = *(_QWORD *)(a1 + 72);
      v43 = v41 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v27 = *(_DWORD *)(a1 + 80) + 1;
      result = v54;
      v25 = (_QWORD *)(v42 + 16LL * v43);
      v44 = *v25;
      if ( a2 != *v25 )
      {
        v45 = 1;
        v46 = 0;
        while ( v44 != -4096 )
        {
          if ( v44 == -8192 && !v46 )
            v46 = v25;
          v43 = v41 & (v45 + v43);
          v25 = (_QWORD *)(v42 + 16LL * v43);
          v44 = *v25;
          if ( a2 == *v25 )
            goto LABEL_28;
          ++v45;
        }
        if ( v46 )
          v25 = v46;
      }
      goto LABEL_28;
    }
    goto LABEL_83;
  }
  if ( v18 - *(_DWORD *)(a1 + 84) - v27 <= v18 >> 3 )
  {
    v55 = result;
    sub_DB2970(a1 + 64, v18);
    v47 = *(_DWORD *)(a1 + 88);
    if ( v47 )
    {
      v48 = v47 - 1;
      v49 = *(_QWORD *)(a1 + 72);
      v50 = 0;
      LODWORD(v51) = v48 & v20;
      v52 = 1;
      v27 = *(_DWORD *)(a1 + 80) + 1;
      result = v55;
      v25 = (_QWORD *)(v49 + 16LL * (unsigned int)v51);
      v53 = *v25;
      if ( a2 != *v25 )
      {
        while ( v53 != -4096 )
        {
          if ( !v50 && v53 == -8192 )
            v50 = v25;
          v51 = v48 & (unsigned int)(v51 + v52);
          v25 = (_QWORD *)(v49 + 16 * v51);
          v53 = *v25;
          if ( a2 == *v25 )
            goto LABEL_28;
          ++v52;
        }
        if ( v50 )
          v25 = v50;
      }
      goto LABEL_28;
    }
LABEL_83:
    ++*(_DWORD *)(a1 + 80);
    BUG();
  }
LABEL_28:
  *(_DWORD *)(a1 + 80) = v27;
  if ( *v25 != -4096 )
    --*(_DWORD *)(a1 + 84);
  *v25 = a2;
  *((_BYTE *)v25 + 8) = result;
  return result;
}
