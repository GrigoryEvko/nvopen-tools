// Function: sub_777100
// Address: 0x777100
//
unsigned __int64 *__fastcall sub_777100(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        unsigned __int64 *a5,
        unsigned __int64 *a6)
{
  __int64 v9; // rax
  unsigned int v10; // ebx
  unsigned __int64 v11; // r9
  unsigned __int64 v12; // r15
  int v13; // r8d
  __int64 v14; // rsi
  unsigned int v15; // edx
  __int64 v16; // rax
  __int64 v17; // rcx
  unsigned int v18; // r14d
  __int64 v19; // rax
  _QWORD *v21; // rax
  unsigned __int64 v22; // rdx
  unsigned __int64 v23; // rsi
  unsigned int v24; // edx
  unsigned __int64 *v25; // rax
  unsigned __int64 v26; // rcx
  unsigned __int64 v27; // rdx
  int v28; // r8d
  unsigned __int64 v29; // r13
  char v30; // al
  unsigned __int64 i; // rdx
  unsigned int v32; // edx
  __int64 v33; // rax
  unsigned __int64 j; // r9
  unsigned int v35; // r9d
  __int64 v36; // rax
  unsigned __int64 v37; // rsi
  unsigned int v38; // edx
  __int64 v39; // [rsp+0h] [rbp-60h]
  _DWORD v42[13]; // [rsp+2Ch] [rbp-34h] BYREF

  if ( *(_BYTE *)(a4 + 140) == 11 )
  {
    v21 = *(_QWORD **)(a2 + 16);
    while ( 1 )
    {
      v21 = (_QWORD *)*v21;
      if ( !v21 )
        break;
      v22 = v21[2];
      if ( v22 && v21[3] == a3 )
      {
        *a5 = v22;
        *a6 = 0;
        return a6;
      }
    }
LABEL_63:
    sub_721090();
  }
  v9 = sub_76FF70(*(_QWORD *)(a4 + 160));
  v42[0] = 1;
  v39 = *(_QWORD *)a2 - a3;
  v10 = *(_DWORD *)a2 - a3;
  if ( v9 )
  {
    v12 = sub_76FF70(*(_QWORD *)(v9 + 112));
    if ( v12 )
    {
      v13 = qword_4F08388;
      v14 = qword_4F08380;
      while ( 1 )
      {
        v15 = v13 & (v12 >> 3);
        v16 = v14 + 16LL * v15;
        v17 = *(_QWORD *)v16;
        if ( v12 == *(_QWORD *)v16 )
        {
LABEL_13:
          v18 = *(_DWORD *)(v16 + 8);
          if ( v10 < v18 )
            goto LABEL_14;
        }
        else
        {
          while ( v17 )
          {
            v15 = v13 & (v15 + 1);
            v16 = v14 + 16LL * v15;
            v17 = *(_QWORD *)v16;
            if ( *(_QWORD *)v16 == v12 )
              goto LABEL_13;
          }
          v18 = 0;
        }
        if ( v10 == v18 && (*(_BYTE *)(a2 + 8) & 2) != 0 )
        {
LABEL_14:
          *a5 = v11;
          *a6 = 0;
          return a6;
        }
        v19 = sub_76FF70(*(_QWORD *)(v12 + 112));
        if ( !v19 )
          goto LABEL_23;
        v12 = v19;
      }
    }
    v12 = v11;
    v18 = 8;
LABEL_23:
    v23 = *(_QWORD *)(v12 + 120);
    v24 = 16;
    if ( (unsigned __int8)(*(_BYTE *)(v23 + 140) - 2) > 1u )
      v24 = sub_7764B0(a1, v23, v42);
    if ( (unsigned int)v39 - v18 < v24 || (_DWORD)v39 - v18 == v24 && ((*(_BYTE *)(a2 + 8) & 2) != 0 || !v24) )
    {
      *a5 = v12;
      *a6 = 0;
      return a6;
    }
  }
  else
  {
    v18 = 8;
  }
  v25 = *(unsigned __int64 **)(a4 + 168);
  v26 = *v25;
  if ( !*v25 )
    BUG();
  v27 = *v25;
  v28 = 0;
  v29 = 0;
LABEL_31:
  while ( 2 )
  {
    v30 = *(_BYTE *)(v27 + 96);
    if ( (v30 & 2) != 0 )
    {
      v27 = *(_QWORD *)v27;
      v28 = 1;
      if ( v27 )
        continue;
LABEL_33:
      if ( v28 )
      {
LABEL_36:
        if ( (*(_BYTE *)(v26 + 96) & 2) == 0 )
          goto LABEL_35;
        for ( i = v26 >> 3; ; LODWORD(i) = v32 + 1 )
        {
          v32 = qword_4F08388 & i;
          v33 = qword_4F08380 + 16LL * v32;
          if ( *(_QWORD *)v33 == v26 )
            break;
          if ( !*(_QWORD *)v33 )
          {
            v29 = v26;
            v18 = 0;
LABEL_35:
            v26 = *(_QWORD *)v26;
            if ( !v26 )
              goto LABEL_55;
            goto LABEL_36;
          }
        }
        v18 = *(_DWORD *)(v33 + 8);
        if ( !v29 || v10 >= v18 )
        {
          v29 = v26;
          goto LABEL_35;
        }
        goto LABEL_60;
      }
LABEL_55:
      v37 = *(_QWORD *)(v29 + 40);
      v38 = 16;
      if ( (unsigned __int8)(*(_BYTE *)(v37 + 140) - 2) > 1u )
        v38 = sub_7764B0(a1, v37, v42);
      if ( (unsigned int)v39 - v18 < v38 || (_DWORD)v39 - v18 == v38 && (*(_BYTE *)(a2 + 8) & 2) != 0 )
        goto LABEL_60;
      goto LABEL_63;
    }
    break;
  }
  if ( (v30 & 1) == 0 )
    goto LABEL_30;
  for ( j = v27 >> 3; ; LODWORD(j) = v35 + 1 )
  {
    v35 = qword_4F08388 & j;
    v36 = qword_4F08380 + 16LL * v35;
    if ( *(_QWORD *)v36 == v27 )
      break;
    if ( !*(_QWORD *)v36 )
    {
      v29 = v27;
      v18 = 0;
LABEL_30:
      v27 = *(_QWORD *)v27;
      if ( !v27 )
        goto LABEL_33;
      goto LABEL_31;
    }
  }
  v18 = *(_DWORD *)(v36 + 8);
  if ( !v29 || v10 >= v18 )
  {
    v29 = v27;
    goto LABEL_30;
  }
LABEL_60:
  *a5 = 0;
  *a6 = v29;
  return a6;
}
