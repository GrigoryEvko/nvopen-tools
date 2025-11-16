// Function: sub_14A6CB0
// Address: 0x14a6cb0
//
__int64 __fastcall sub_14A6CB0(__int64 a1, __int64 a2, __int64 a3, __int64 *a4, bool *a5)
{
  unsigned int v5; // r13d
  __int64 v9; // rdx
  char *v10; // r14
  __int64 v11; // rdi
  char v12; // al
  char *v13; // rsi
  bool v14; // al
  __int64 v15; // rax
  __int64 v16; // rax
  _QWORD *v17; // r9
  _BYTE *v18; // r12
  int v19; // eax
  unsigned int v20; // ecx
  __int64 v21; // rsi
  int v22; // r8d
  unsigned int v23; // eax
  __int64 v24; // r10
  _QWORD *v25; // rdx
  unsigned int v26; // eax
  __int64 v27; // rcx
  _QWORD *v28; // rdx
  __int64 v30; // rax
  _QWORD *v31; // rdx
  unsigned int v32; // edx
  __int64 v33; // rcx
  _QWORD *v34; // rax
  bool v35; // bl
  __int64 v36; // rax
  __int64 v37; // r11
  __int64 v38; // [rsp+0h] [rbp-50h]
  unsigned __int8 v39; // [rsp+0h] [rbp-50h]
  __int64 v40; // [rsp+8h] [rbp-48h]

  v9 = *(unsigned int *)(a1 + 8);
  v40 = a2;
  v10 = *(char **)(a1 + 8 * (1 - v9));
  v11 = *(_QWORD *)(a1 - 8 * v9);
  if ( v10 && (v12 = *v10, (unsigned __int8)(*v10 - 4) <= 0x1Eu) )
  {
    v13 = v10;
    if ( !v11 )
    {
      v5 = 0;
      if ( (unsigned int)v9 <= 3 )
        goto LABEL_13;
      goto LABEL_10;
    }
  }
  else
  {
    v13 = 0;
    if ( !v11 )
    {
      v14 = 1;
      goto LABEL_6;
    }
  }
  if ( (unsigned __int8)(*(_BYTE *)v11 - 4) > 0x1Eu )
  {
    v14 = v13 == 0;
    v11 = 0;
  }
  else
  {
    v14 = v13 == (char *)v11;
  }
LABEL_6:
  LOBYTE(v5) = v14 && a3 == (_QWORD)v13;
  if ( (_BYTE)v5 )
  {
    if ( a4 )
      *a4 = sub_14A6B70(a3);
    goto LABEL_69;
  }
  if ( (unsigned int)v9 > 3 )
  {
    if ( !v10 )
    {
      v5 = 1;
      goto LABEL_13;
    }
    v12 = *v10;
LABEL_10:
    v5 = 1;
    if ( (unsigned __int8)(v12 - 4) <= 0x1Eu )
    {
      v15 = *((unsigned int *)v10 + 2);
      v5 = 0;
      if ( (unsigned int)v15 > 2 )
        LOBYTE(v5) = (unsigned __int8)(**(_BYTE **)&v10[-8 * v15] - 4) <= 0x1Eu;
    }
  }
LABEL_13:
  v16 = *(_QWORD *)(*(_QWORD *)(a1 + 8 * (2 - v9)) + 136LL);
  v17 = *(_QWORD **)(v16 + 24);
  if ( *(_DWORD *)(v16 + 32) > 0x40u )
    v17 = (_QWORD *)*v17;
  if ( v11 )
  {
    v38 = *(unsigned int *)(a2 + 8);
    v18 = *(_BYTE **)(a2 - 8 * v38);
    while ( 1 )
    {
      if ( v18 )
      {
        v19 = (unsigned __int8)*v18 - 4;
        if ( v11 == (_QWORD)v18 && (unsigned __int8)(*v18 - 4) <= 0x1Eu )
        {
          LOBYTE(v19) = v11 == (_QWORD)v18 && (unsigned __int8)(*v18 - 4) <= 0x1Eu;
          v32 = v19;
          v33 = *(_QWORD *)(*(_QWORD *)(a2 + 8 * (2 - v38)) + 136LL);
          v34 = *(_QWORD **)(v33 + 24);
          if ( *(_DWORD *)(v33 + 32) > 0x40u )
            v34 = (_QWORD *)*v34;
          v35 = v34 == v17;
          if ( a4 )
          {
            if ( v34 != v17 )
            {
              v39 = v32;
              v36 = sub_14A6B70(a3);
              v32 = v39;
              v40 = v36;
            }
            *a4 = v40;
          }
          v5 = v32;
          *a5 = v35;
          return v5;
        }
      }
      if ( v10 && (_BYTE)v5 && (unsigned __int8)(*v10 - 4) <= 0x1Eu && (char *)v11 == v10 )
        goto LABEL_63;
      v20 = *(_DWORD *)(v11 + 8);
      if ( v20 <= 2 )
        break;
      v21 = v20;
      if ( (unsigned __int8)(**(_BYTE **)(v11 - 8LL * v20) - 4) <= 0x1Eu )
      {
        if ( v20 <= 5 )
          goto LABEL_36;
        v22 = 3;
        v23 = 3;
        while ( 1 )
        {
LABEL_28:
          v24 = *(_QWORD *)(*(_QWORD *)(v11 + 8 * (v23 + 1 - (unsigned __int64)v20)) + 136LL);
          v25 = *(_QWORD **)(v24 + 24);
          if ( *(_DWORD *)(v24 + 32) > 0x40u )
            v25 = (_QWORD *)*v25;
          if ( v25 > v17 )
            break;
          v23 += v22;
          if ( v23 >= v20 )
            goto LABEL_46;
        }
        v26 = v23 - v22;
        if ( v26 )
          goto LABEL_32;
LABEL_46:
        v26 = v20 - v22;
LABEL_32:
        v27 = *(_QWORD *)(*(_QWORD *)(v11 + 8 * (v26 + 1 - (unsigned __int64)v20)) + 136LL);
        v28 = *(_QWORD **)(v27 + 24);
        if ( *(_DWORD *)(v27 + 32) > 0x40u )
          v28 = (_QWORD *)*v28;
        v11 = *(_QWORD *)(v11 + 8 * (v26 - v21));
        if ( !v11 || (unsigned __int8)(*(_BYTE *)v11 - 4) > 0x1Eu )
          goto LABEL_36;
        v17 = (_QWORD *)((char *)v17 - (__int64)v28);
      }
      else
      {
        if ( v20 != 3 )
        {
          v22 = 2;
          v23 = 1;
          goto LABEL_28;
        }
        v30 = *(_QWORD *)(*(_QWORD *)(v11 - 8) + 136LL);
        v31 = *(_QWORD **)(v30 + 24);
        if ( *(_DWORD *)(v30 + 32) <= 0x40u )
          v17 = (_QWORD *)((char *)v17 - (__int64)v31);
        else
          v17 = (_QWORD *)((char *)v17 - *v31);
LABEL_50:
        v11 = *(_QWORD *)(v11 + 8 * (1 - v21));
        if ( !v11 || (unsigned __int8)(*(_BYTE *)v11 - 4) > 0x1Eu )
          goto LABEL_36;
      }
    }
    if ( v20 != 2 )
      goto LABEL_36;
    v21 = 2;
    goto LABEL_50;
  }
LABEL_36:
  if ( (_BYTE)v5 )
  {
    v11 = 0;
    v18 = *(_BYTE **)(a2 - 8LL * *(unsigned int *)(a2 + 8));
LABEL_63:
    if ( v18 && (unsigned __int8)(*v18 - 4) >= 0x1Fu )
      v18 = 0;
    v5 = sub_14A6A50(v11, (__int64)v18);
    if ( (_BYTE)v5 )
    {
      if ( a4 )
        *a4 = sub_14A6B70(v37);
LABEL_69:
      *a5 = 1;
    }
  }
  return v5;
}
