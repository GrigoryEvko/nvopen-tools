// Function: sub_BFA370
// Address: 0xbfa370
//
void __fastcall sub_BFA370(__int64 *a1, __int64 a2)
{
  int v4; // edi
  __int64 v5; // rcx
  __int64 v6; // rsi
  int v7; // edx
  const char *v8; // rax
  __int64 v9; // r14
  _BYTE *v10; // rax
  __int64 v11; // rax
  __int64 v12; // r8
  int v13; // edx
  char v14; // al
  _QWORD *v15; // r14
  __int64 v16; // r15
  char *v17; // rax
  int v18; // r8d
  __int64 v19; // rsi
  __int64 v20; // r8
  _QWORD *v21; // rdi
  __int64 v22; // rdx
  _QWORD *v23; // rax
  __int64 v24; // r10
  int v25; // r9d
  __int64 v26; // r10
  int v27; // r9d
  __int64 v28; // r10
  int v29; // r9d
  __int64 v30; // r10
  int v31; // r9d
  __int64 v32; // r14
  __int64 v33; // rcx
  int v34; // esi
  __int64 v35; // rdi
  char v36; // dl
  __int64 v37; // r8
  _BYTE *v38; // rax
  const char *v39; // rdi
  __int64 v40; // r14
  _BYTE *v41; // rax
  __int64 v42; // rax
  __int64 v43; // r9
  int v44; // ecx
  __int64 v45; // rdx
  int v46; // r9d
  bool v47; // r8
  int v48; // ecx
  const char *v49; // rax
  __int64 v50; // r9
  int v51; // ecx
  __int64 v52; // r9
  int v53; // ecx
  char v54; // cl
  const char *v55; // rax
  const char *v56; // r14
  __int64 v57; // rsi
  int v58; // r10d
  _BYTE *v59; // [rsp+8h] [rbp-F8h] BYREF
  _QWORD v60[4]; // [rsp+10h] [rbp-F0h] BYREF
  char v61; // [rsp+30h] [rbp-D0h]
  char v62; // [rsp+31h] [rbp-CFh]
  const char *v63; // [rsp+40h] [rbp-C0h] BYREF
  __int64 v64; // [rsp+48h] [rbp-B8h]
  _BYTE v65[17]; // [rsp+50h] [rbp-B0h] BYREF
  char v66; // [rsp+61h] [rbp-9Fh]

  v4 = *(_DWORD *)(a2 + 4);
  v5 = v4 & 0x7FFFFFF;
  v6 = *(_QWORD *)(*(_QWORD *)(a2 - 32 * v5) + 8LL);
  v7 = *(unsigned __int8 *)(v6 + 8);
  if ( (unsigned int)(v7 - 17) <= 1 )
    LOBYTE(v7) = *(_BYTE *)(**(_QWORD **)(v6 + 16) + 8LL);
  if ( (_BYTE)v7 != 14 )
  {
    v66 = 1;
    v8 = "GEP base pointer is not a vector or a vector of pointers";
    goto LABEL_5;
  }
  v12 = *(_QWORD *)(a2 + 72);
  v13 = *(unsigned __int8 *)(v12 + 8);
  v14 = *(_BYTE *)(v12 + 8);
  if ( (_BYTE)v13 == 12 )
  {
LABEL_16:
    v63 = v65;
    v64 = 0x1000000000LL;
    v15 = (_QWORD *)(a2 + 32 * (1 - v5));
    v16 = (-32 * (1 - v5)) >> 5;
    if ( (unsigned __int64)(-32 * (1 - v5)) > 0x200 )
    {
      sub_C8D5F0(&v63, v65, v16, 8);
      v19 = (__int64)v63;
      v18 = v64;
      v17 = (char *)&v63[8 * (unsigned int)v64];
    }
    else
    {
      v17 = v65;
      v18 = 0;
      v19 = (__int64)v65;
    }
    if ( (_QWORD *)a2 != v15 )
    {
      do
      {
        if ( v17 )
          *(_QWORD *)v17 = *v15;
        v15 += 4;
        v17 += 8;
      }
      while ( (_QWORD *)a2 != v15 );
      v19 = (__int64)v63;
      v18 = v64;
    }
    LODWORD(v64) = v16 + v18;
    v20 = (unsigned int)(v16 + v18);
    v21 = (_QWORD *)(v19 + 8 * v20);
    v22 = (8 * v20) >> 3;
    if ( (8 * v20) >> 5 )
    {
      v23 = (_QWORD *)v19;
      while ( 1 )
      {
        v30 = *(_QWORD *)(*v23 + 8LL);
        v31 = *(unsigned __int8 *)(v30 + 8);
        if ( (unsigned int)(v31 - 17) <= 1 )
          LOBYTE(v31) = *(_BYTE *)(**(_QWORD **)(v30 + 16) + 8LL);
        if ( (_BYTE)v31 != 12 )
          goto LABEL_38;
        v24 = *(_QWORD *)(v23[1] + 8LL);
        v25 = *(unsigned __int8 *)(v24 + 8);
        if ( (unsigned int)(v25 - 17) <= 1 )
          LOBYTE(v25) = *(_BYTE *)(**(_QWORD **)(v24 + 16) + 8LL);
        if ( (_BYTE)v25 != 12 )
        {
          ++v23;
          goto LABEL_38;
        }
        v26 = *(_QWORD *)(v23[2] + 8LL);
        v27 = *(unsigned __int8 *)(v26 + 8);
        if ( (unsigned int)(v27 - 17) <= 1 )
          LOBYTE(v27) = *(_BYTE *)(**(_QWORD **)(v26 + 16) + 8LL);
        if ( (_BYTE)v27 != 12 )
        {
          v23 += 2;
          goto LABEL_38;
        }
        v28 = *(_QWORD *)(v23[3] + 8LL);
        v29 = *(unsigned __int8 *)(v28 + 8);
        if ( (unsigned int)(v29 - 17) <= 1 )
          LOBYTE(v29) = *(_BYTE *)(**(_QWORD **)(v28 + 16) + 8LL);
        if ( (_BYTE)v29 != 12 )
          break;
        v23 += 4;
        if ( (_QWORD *)(v19 + 32 * ((8 * v20) >> 5)) == v23 )
        {
          v22 = v21 - v23;
          goto LABEL_72;
        }
      }
      v23 += 3;
LABEL_38:
      if ( v21 != v23 )
      {
        v40 = *a1;
        v62 = 1;
        v60[0] = "GEP indexes must be integers";
        v61 = 3;
        if ( v40 )
        {
          v19 = v40;
          sub_CA0E80(v60, v40);
          v41 = *(_BYTE **)(v40 + 32);
          if ( (unsigned __int64)v41 >= *(_QWORD *)(v40 + 24) )
          {
            v19 = 10;
            sub_CB5D20(v40, 10);
          }
          else
          {
            *(_QWORD *)(v40 + 32) = v41 + 1;
            *v41 = 10;
          }
          v42 = *a1;
          *((_BYTE *)a1 + 152) = 1;
          if ( v42 )
          {
            v19 = a2;
            sub_BDBD80((__int64)a1, (_BYTE *)a2);
          }
        }
        else
        {
          *((_BYTE *)a1 + 152) = 1;
        }
LABEL_48:
        v39 = v63;
        if ( v63 == v65 )
          return;
        goto LABEL_49;
      }
      goto LABEL_39;
    }
    v23 = (_QWORD *)v19;
LABEL_72:
    if ( v22 != 2 )
    {
      if ( v22 != 3 )
      {
        if ( v22 != 1 )
          goto LABEL_39;
        goto LABEL_75;
      }
      v50 = *(_QWORD *)(*v23 + 8LL);
      v51 = *(unsigned __int8 *)(v50 + 8);
      if ( (unsigned int)(v51 - 17) <= 1 )
        LOBYTE(v51) = *(_BYTE *)(**(_QWORD **)(v50 + 16) + 8LL);
      if ( (_BYTE)v51 != 12 )
        goto LABEL_38;
      ++v23;
    }
    v52 = *(_QWORD *)(*v23 + 8LL);
    v53 = *(unsigned __int8 *)(v52 + 8);
    if ( (unsigned int)(v53 - 17) <= 1 )
      LOBYTE(v53) = *(_BYTE *)(**(_QWORD **)(v52 + 16) + 8LL);
    if ( (_BYTE)v53 != 12 )
      goto LABEL_38;
    ++v23;
LABEL_75:
    v43 = *(_QWORD *)(*v23 + 8LL);
    v44 = *(unsigned __int8 *)(v43 + 8);
    if ( (unsigned int)(v44 - 17) <= 1 )
      LOBYTE(v44) = *(_BYTE *)(**(_QWORD **)(v43 + 16) + 8LL);
    if ( (_BYTE)v44 != 12 )
      goto LABEL_38;
LABEL_39:
    v32 = sub_B4DC50(*(_QWORD *)(a2 + 72), v19, v20);
    if ( !v32 )
    {
      v59 = (_BYTE *)a2;
      v49 = "Invalid indices for GEP pointer type!";
      v62 = 1;
      goto LABEL_84;
    }
    v33 = *(_QWORD *)(a2 + 8);
    v34 = *(unsigned __int8 *)(v33 + 8);
    if ( (unsigned int)(v34 - 17) > 1 )
    {
      v36 = *(_BYTE *)(v33 + 8);
      v35 = *(_QWORD *)(a2 + 8);
    }
    else
    {
      v35 = **(_QWORD **)(v33 + 16);
      v36 = *(_BYTE *)(v35 + 8);
    }
    if ( v36 != 14 || v32 != *(_QWORD *)(a2 + 80) )
    {
      v19 = (__int64)v60;
      v62 = 1;
      v60[0] = "GEP is not of right type for indices!";
      v61 = 3;
      sub_BDBF70(a1, (__int64)v60);
      if ( *a1 )
      {
        sub_BDBD80((__int64)a1, (_BYTE *)a2);
        v37 = *a1;
        v38 = *(_BYTE **)(*a1 + 32);
        if ( (unsigned __int64)v38 >= *(_QWORD *)(*a1 + 24) )
        {
          v37 = sub_CB5D20(*a1, 32);
        }
        else
        {
          *(_QWORD *)(v37 + 32) = v38 + 1;
          *v38 = 32;
        }
        v19 = v37;
        sub_A587F0(v32, v37, 0, 0);
      }
      goto LABEL_48;
    }
    v45 = *(_QWORD *)(*(_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)) + 8LL);
    if ( (unsigned int)(v34 - 17) > 1 )
    {
LABEL_98:
      if ( (unsigned int)*(unsigned __int8 *)(v45 + 8) - 17 > 1 )
      {
LABEL_100:
        if ( *(_DWORD *)(v35 + 8) >> 8 == *(_DWORD *)(v45 + 8) >> 8 )
        {
          v19 = a2;
          sub_BF6FE0((__int64)a1, a2);
          v39 = v63;
          if ( v63 == v65 )
            return;
LABEL_49:
          _libc_free(v39, v19);
          return;
        }
        v59 = (_BYTE *)a2;
        v49 = "GEP address space doesn't match type";
        v62 = 1;
LABEL_84:
        v19 = (__int64)v60;
        v60[0] = v49;
        v61 = 3;
        sub_BEF6C0(a1, (__int64)v60, &v59);
        goto LABEL_48;
      }
LABEL_99:
      v45 = **(_QWORD **)(v45 + 16);
      goto LABEL_100;
    }
    v46 = *(_DWORD *)(v33 + 32);
    v47 = (_BYTE)v34 == 18;
    v48 = *(unsigned __int8 *)(v45 + 8);
    if ( (unsigned int)(v48 - 17) > 1 )
    {
      v55 = v63;
      v56 = &v63[8 * (unsigned int)v64];
      if ( v63 != v56 )
        goto LABEL_107;
      goto LABEL_100;
    }
    if ( ((_BYTE)v48 == 18) != v47 || *(_DWORD *)(v45 + 32) != v46 )
    {
      v59 = (_BYTE *)a2;
      v49 = "Vector GEP result width doesn't match operand's";
      v62 = 1;
      goto LABEL_84;
    }
    v55 = v63;
    v56 = &v63[8 * (unsigned int)v64];
    if ( v63 == v56 )
      goto LABEL_99;
    while ( 1 )
    {
LABEL_107:
      v57 = *(_QWORD *)(*(_QWORD *)v55 + 8LL);
      v58 = *(unsigned __int8 *)(v57 + 8);
      v54 = *(_BYTE *)(v57 + 8);
      if ( (unsigned int)(v58 - 17) > 1 )
      {
        if ( v58 != 18 )
          goto LABEL_96;
      }
      else if ( *(_DWORD *)(v57 + 32) != v46 || ((_BYTE)v58 == 18) != v47 )
      {
        v59 = (_BYTE *)a2;
        v49 = "Invalid GEP index vector width";
        v62 = 1;
        goto LABEL_84;
      }
      v54 = *(_BYTE *)(**(_QWORD **)(v57 + 16) + 8LL);
LABEL_96:
      if ( v54 != 12 )
      {
        v19 = (__int64)v60;
        v62 = 1;
        v60[0] = "All GEP indices should be of integer type";
        v61 = 3;
        sub_BDBF70(a1, (__int64)v60);
        goto LABEL_48;
      }
      v55 += 8;
      if ( v56 == v55 )
        goto LABEL_98;
    }
  }
  if ( (unsigned __int8)v13 > 3u )
  {
    if ( (_BYTE)v13 == 5 )
    {
LABEL_59:
      v5 = v4 & 0x7FFFFFF;
      goto LABEL_16;
    }
    if ( (v13 & 0xFB) != 0xA && (v13 & 0xFD) != 4 )
    {
      if ( (unsigned __int8)(v14 - 15) > 3u && v13 != 20 || !(unsigned __int8)sub_BCEBA0(*(_QWORD *)(a2 + 72), 0) )
      {
        v66 = 1;
        v8 = "GEP into unsized type!";
        goto LABEL_5;
      }
      v12 = *(_QWORD *)(a2 + 72);
      v14 = *(_BYTE *)(v12 + 8);
    }
  }
  if ( v14 != 15 )
  {
    v4 = *(_DWORD *)(a2 + 4);
    goto LABEL_59;
  }
  if ( !sub_BCEA30(v12) )
  {
    v5 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
    goto LABEL_16;
  }
  v66 = 1;
  v8 = "getelementptr cannot target structure that contains scalable vectortype";
LABEL_5:
  v9 = *a1;
  v63 = v8;
  v65[16] = 3;
  if ( v9 )
  {
    sub_CA0E80(&v63, v9);
    v10 = *(_BYTE **)(v9 + 32);
    if ( (unsigned __int64)v10 >= *(_QWORD *)(v9 + 24) )
    {
      sub_CB5D20(v9, 10);
    }
    else
    {
      *(_QWORD *)(v9 + 32) = v10 + 1;
      *v10 = 10;
    }
    v11 = *a1;
    *((_BYTE *)a1 + 152) = 1;
    if ( v11 )
      sub_BDBD80((__int64)a1, (_BYTE *)a2);
  }
  else
  {
    *((_BYTE *)a1 + 152) = 1;
  }
}
