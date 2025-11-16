// Function: sub_2FD64C0
// Address: 0x2fd64c0
//
__int64 __fastcall sub_2FD64C0(__int64 *a1, unsigned __int8 a2, __int64 *a3)
{
  __int64 *v3; // r15
  __int64 v4; // r12
  int v5; // ebx
  char v6; // al
  __int64 v7; // rdi
  __int64 (*v8)(); // rax
  __int64 v9; // r13
  unsigned __int64 v10; // r14
  int v11; // eax
  __int64 v12; // rdi
  int v13; // edx
  unsigned __int64 v14; // rdi
  int v15; // edx
  __int64 i; // r14
  _BYTE *v17; // rax
  _BYTE *v18; // rcx
  __int64 v19; // rdi
  int v20; // r14d
  __int64 v22; // rbx
  __int64 v23; // r13
  __int64 *v24; // r12
  unsigned int v25; // r15d
  __int64 v26; // rax
  int *v27; // rdx
  int v28; // eax
  unsigned __int64 v29; // rax
  __int64 v30; // rcx
  int v31; // eax
  __int64 v32; // rax
  int v33; // edx
  __int64 v34; // r9
  __int64 v35; // r8
  __int64 v36; // rcx
  __int64 v37; // r10
  _DWORD *v38; // rdi
  int v39; // esi
  __int64 v40; // rax
  __int64 v41; // rax
  int v42; // eax
  __int64 v43; // rax
  int v44; // eax
  __int64 v45; // rax
  char v46; // al
  __int64 v47; // rax
  __int64 v48; // rax
  __int64 v49; // rsi
  __int64 v50; // rax
  __int64 *v51; // r8
  __int64 *v52; // rbx
  __int64 v53; // r8
  __int64 v54; // r8
  __int64 v55; // r8
  __int64 v56; // rax
  unsigned __int8 v57; // [rsp+Ah] [rbp-106h]
  bool v58; // [rsp+Bh] [rbp-105h]
  __int64 v60; // [rsp+10h] [rbp-100h]
  unsigned __int8 v61; // [rsp+1Bh] [rbp-F5h]
  unsigned int v62; // [rsp+1Ch] [rbp-F4h]
  __int64 v63; // [rsp+20h] [rbp-F0h] BYREF
  __int64 v64; // [rsp+28h] [rbp-E8h] BYREF
  unsigned __int64 v65[2]; // [rsp+30h] [rbp-E0h] BYREF
  _BYTE v66[208]; // [rsp+40h] [rbp-D0h] BYREF

  v3 = a1;
  v4 = (__int64)a3;
  if ( !*((_BYTE *)a1 + 57) && sub_2E32580(a3) )
    return 0;
  v61 = sub_2E322C0(v4, v4);
  if ( v61 )
    return 0;
  v5 = *((_DWORD *)a1 + 15);
  if ( !v5 )
    v5 = qword_5026908;
  v6 = sub_2EE6AD0(v4, a1[6], (__int64 **)a1[5]);
  v7 = *a1;
  v63 = 0;
  v64 = 0;
  if ( v6 )
    v5 = 1;
  v65[0] = (unsigned __int64)v66;
  v65[1] = 0x400000000LL;
  v8 = *(__int64 (**)())(*(_QWORD *)v7 + 344LL);
  if ( (v8 == sub_2DB1AE0
     || ((unsigned __int8 (__fastcall *)(__int64, __int64, __int64 *, __int64 *, unsigned __int64 *, _QWORD))v8)(
          v7,
          v4,
          &v63,
          &v64,
          v65,
          0))
    && sub_2E32580((__int64 *)v4) )
  {
    goto LABEL_38;
  }
  v9 = v4 + 48;
  v10 = *(_QWORD *)(v4 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v4 + 48 == v10 )
  {
    v19 = *(_QWORD *)(v4 + 56);
    v57 = 0;
    if ( v9 == v19 )
    {
      v20 = 0;
      goto LABEL_123;
    }
    v58 = 0;
    goto LABEL_33;
  }
  if ( !v10 )
    BUG();
  v11 = *(_DWORD *)(v10 + 44);
  v12 = *(_QWORD *)v10;
  LOBYTE(v13) = v11;
  if ( (*(_QWORD *)v10 & 4) != 0 )
  {
    v14 = *(_QWORD *)(v4 + 48) & 0xFFFFFFFFFFFFFFF8LL;
    if ( (v11 & 4) != 0 )
    {
LABEL_14:
      v57 = (*(_QWORD *)(*(_QWORD *)(v14 + 16) + 24LL) & 0x800LL) != 0;
      goto LABEL_15;
    }
  }
  else if ( (v11 & 4) != 0 )
  {
    while ( 1 )
    {
      v14 = v12 & 0xFFFFFFFFFFFFFFF8LL;
      v13 = *(_DWORD *)(v14 + 44) & 0xFFFFFF;
      if ( (*(_DWORD *)(v14 + 44) & 4) == 0 )
        break;
      v12 = *(_QWORD *)v14;
    }
  }
  else
  {
    v14 = *(_QWORD *)(v4 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  }
  if ( (v13 & 8) == 0 )
    goto LABEL_14;
  v57 = sub_2E88A90(v14, 2048, 1);
  v10 = *(_QWORD *)(v4 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( !v10 )
    BUG();
LABEL_15:
  v15 = *(_DWORD *)(v10 + 44);
  if ( (*(_QWORD *)v10 & 4) != 0 )
  {
    if ( (v15 & 4) != 0 )
      goto LABEL_118;
  }
  else if ( (v15 & 4) != 0 )
  {
    for ( i = *(_QWORD *)v10; ; i = *(_QWORD *)v10 )
    {
      v10 = i & 0xFFFFFFFFFFFFFFF8LL;
      v15 = *(_DWORD *)(v10 + 44) & 0xFFFFFF;
      if ( (*(_DWORD *)(v10 + 44) & 4) == 0 )
        break;
    }
  }
  if ( (v15 & 8) != 0 )
  {
    v58 = sub_2E88A90(v10, 2048, 1);
    goto LABEL_22;
  }
LABEL_118:
  v58 = (*(_QWORD *)(*(_QWORD *)(v10 + 16) + 24LL) & 0x800LL) != 0;
LABEL_22:
  if ( v58 )
  {
    v17 = sub_2FD5B50(*(_BYTE **)(v10 + 32), *(_QWORD *)(v10 + 32) + 40LL * (*(_DWORD *)(v10 + 40) & 0xFFFFFF));
    v58 = v18 == v17;
  }
  if ( v57 && *((_BYTE *)v3 + 56) )
  {
    v57 = *((_BYTE *)v3 + 56);
    v5 = qword_5026828;
  }
  v19 = *(_QWORD *)(v4 + 56);
  if ( v9 == v19 )
  {
    v20 = 0;
    goto LABEL_56;
  }
LABEL_33:
  v62 = v5;
  v20 = 0;
  v22 = v19;
  v60 = v4 + 48;
  v23 = v4;
  v24 = v3;
  v25 = 0;
  do
  {
    v26 = *(_QWORD *)(v22 + 48);
    v27 = (int *)(v26 & 0xFFFFFFFFFFFFFFF8LL);
    if ( (v26 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
    {
      v28 = v26 & 7;
      switch ( v28 )
      {
        case 1:
          goto LABEL_36;
        case 3:
          v41 = *((unsigned __int8 *)v27 + 4);
          if ( (_BYTE)v41 && *(_QWORD *)&v27[2 * *v27 + 4]
            || *((_BYTE *)v27 + 5) && *(_QWORD *)&v27[2 * *v27 + 4 + 2 * v41] )
          {
            goto LABEL_36;
          }
          break;
        case 2:
          goto LABEL_36;
      }
    }
    v42 = *(_DWORD *)(v22 + 44);
    if ( (v42 & 4) == 0 && (v42 & 8) != 0 )
      LOBYTE(v43) = sub_2E88A90(v22, 0x800000, 1);
    else
      v43 = (*(_QWORD *)(*(_QWORD *)(v22 + 16) + 24LL) >> 23) & 1LL;
    if ( !(_BYTE)v43 )
    {
      if ( (unsigned int)*(unsigned __int16 *)(v22 + 68) - 1 <= 1
        && (*(_BYTE *)(*(_QWORD *)(v22 + 32) + 64LL) & 0x20) != 0 )
      {
        goto LABEL_38;
      }
      goto LABEL_41;
    }
LABEL_36:
    v29 = *(unsigned int *)(*(_QWORD *)(*(_QWORD *)(v23 + 32) + 8LL) + 556LL);
    if ( (unsigned int)v29 <= 0x1F )
    {
      v30 = 3623879202LL;
      if ( _bittest64(&v30, v29) )
        goto LABEL_38;
    }
    if ( *(_WORD *)(v22 + 68) != 3 )
      goto LABEL_38;
LABEL_41:
    v31 = *(_DWORD *)(v22 + 44);
    if ( (v31 & 0x20000) == 0 )
    {
      if ( (v31 & 4) != 0 || (v31 & 8) == 0 )
        v32 = (*(_QWORD *)(*(_QWORD *)(v22 + 16) + 24LL) >> 36) & 1LL;
      else
        LOBYTE(v32) = sub_2E88A90(v22, 0x1000000000LL, 1);
      if ( (_BYTE)v32 )
        goto LABEL_38;
    }
    if ( *((_BYTE *)v24 + 56) )
    {
      v44 = *(_DWORD *)(v22 + 44);
      if ( (v44 & 4) != 0 )
      {
        v45 = *(_QWORD *)(v22 + 16);
        if ( (*(_BYTE *)(v45 + 24) & 0x20) != 0 )
          goto LABEL_38;
        goto LABEL_95;
      }
      if ( (v44 & 8) != 0 )
      {
        if ( sub_2E88A90(v22, 32, 1) )
          goto LABEL_38;
        if ( !*((_BYTE *)v24 + 56) )
          goto LABEL_47;
        if ( (*(_DWORD *)(v22 + 44) & 4) != 0 || (*(_DWORD *)(v22 + 44) & 8) == 0 )
          goto LABEL_94;
      }
      else
      {
        if ( (*(_BYTE *)(*(_QWORD *)(v22 + 16) + 24LL) & 0x20) != 0 )
          goto LABEL_38;
        if ( (v44 & 8) == 0 )
        {
LABEL_94:
          v45 = *(_QWORD *)(v22 + 16);
LABEL_95:
          v46 = (unsigned __int8)*(_QWORD *)(v45 + 24) >> 7;
          goto LABEL_96;
        }
      }
      v46 = sub_2E88A90(v22, 128, 1);
LABEL_96:
      if ( v46 )
        goto LABEL_38;
    }
LABEL_47:
    v33 = *(unsigned __int16 *)(v22 + 68);
    if ( (_WORD)v33 == 2 )
      goto LABEL_38;
    if ( (_WORD)v33 == 21 )
    {
      v25 += sub_2E89C40(v22);
    }
    else if ( v33 != 68 && *(_WORD *)(v22 + 68) )
    {
      v25 += (*(_QWORD *)(*(_QWORD *)(v22 + 16) + 24LL) & 0x10LL) == 0;
    }
    if ( v25 > v62 )
      goto LABEL_38;
    v20 += *(_WORD *)(v22 + 68) == 0 || *(_WORD *)(v22 + 68) == 68;
    if ( (*(_BYTE *)v22 & 4) == 0 )
    {
      while ( (*(_BYTE *)(v22 + 44) & 8) != 0 )
        v22 = *(_QWORD *)(v22 + 8);
    }
    v22 = *(_QWORD *)(v22 + 8);
  }
  while ( v60 != v22 );
  v3 = v24;
  v4 = v23;
LABEL_56:
  if ( v58 )
  {
    v34 = *(_QWORD *)(v4 + 112);
    v35 = v34 + 8LL * *(unsigned int *)(v4 + 120);
    goto LABEL_58;
  }
LABEL_123:
  v47 = *(unsigned int *)(v4 + 120);
  if ( *(_DWORD *)(v4 + 72) <= (unsigned int)dword_5026748 || dword_5026668 >= (unsigned int)v47 )
  {
    v34 = *(_QWORD *)(v4 + 112);
    v35 = v34 + 8 * v47;
LABEL_58:
    while ( v35 != v34 )
    {
      v36 = *(_QWORD *)(*(_QWORD *)v34 + 56LL);
      v37 = *(_QWORD *)v34 + 48LL;
      if ( v36 != v37 )
      {
        while ( *(_WORD *)(v36 + 68) == 68 || !*(_WORD *)(v36 + 68) )
        {
          v38 = *(_DWORD **)(v36 + 32);
          v39 = *(_DWORD *)(v36 + 40) & 0xFFFFFF;
          if ( v39 != 1 )
          {
            v40 = 1;
            while ( v4 != *(_QWORD *)&v38[10 * (unsigned int)(v40 + 1) + 6] )
            {
              v40 = (unsigned int)(v40 + 2);
              if ( v39 == (_DWORD)v40 )
                goto LABEL_69;
            }
            v38 += 10 * v40;
          }
LABEL_69:
          if ( (*v38 & 0xFFF00) != 0 )
            goto LABEL_38;
          if ( (*(_BYTE *)v36 & 4) == 0 )
          {
            while ( (*(_BYTE *)(v36 + 44) & 8) != 0 )
              v36 = *(_QWORD *)(v36 + 8);
          }
          v36 = *(_QWORD *)(v36 + 8);
          if ( v37 == v36 )
            break;
        }
      }
      v34 += 8;
    }
    v61 = v57;
    if ( !v57 )
    {
      v61 = a2;
      if ( !a2 )
      {
        if ( *((_BYTE *)v3 + 56) )
          v61 = sub_2FD63C0(v3, v4);
        else
          v61 = 1;
      }
    }
    goto LABEL_38;
  }
  if ( v20 )
    goto LABEL_38;
  v34 = *(_QWORD *)(v4 + 112);
  v48 = 8 * v47;
  v49 = v34 + v48;
  v50 = v48 >> 5;
  if ( !v50 )
  {
    v51 = *(__int64 **)(v4 + 112);
LABEL_142:
    v56 = v49 - (_QWORD)v51;
    if ( v49 - (_QWORD)v51 != 16 )
    {
      if ( v56 != 24 )
      {
        if ( v56 != 8 )
        {
LABEL_145:
          v35 = v49;
          goto LABEL_58;
        }
LABEL_153:
        if ( sub_2FD5D70(*v51) )
          goto LABEL_133;
        goto LABEL_145;
      }
      if ( sub_2FD5D70(*v51) )
        goto LABEL_133;
      v51 = (__int64 *)(v35 + 8);
    }
    if ( sub_2FD5D70(*v51) )
      goto LABEL_133;
    v51 = (__int64 *)(v35 + 8);
    goto LABEL_153;
  }
  v51 = *(__int64 **)(v4 + 112);
  v52 = (__int64 *)(v34 + 32 * v50);
  while ( !sub_2FD5D70(*v51) )
  {
    if ( sub_2FD5D70(*(_QWORD *)(v35 + 8)) )
    {
      v35 = v53 + 8;
      break;
    }
    if ( sub_2FD5D70(*(_QWORD *)(v53 + 16)) )
    {
      v35 = v54 + 16;
      break;
    }
    if ( sub_2FD5D70(*(_QWORD *)(v54 + 24)) )
    {
      v35 = v55 + 24;
      break;
    }
    v51 = (__int64 *)(v55 + 32);
    if ( v52 == v51 )
      goto LABEL_142;
  }
LABEL_133:
  if ( v49 == v35 )
    goto LABEL_58;
LABEL_38:
  if ( (_BYTE *)v65[0] != v66 )
    _libc_free(v65[0]);
  return v61;
}
