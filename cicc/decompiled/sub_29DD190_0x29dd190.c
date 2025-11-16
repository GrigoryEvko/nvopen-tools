// Function: sub_29DD190
// Address: 0x29dd190
//
__int64 __fastcall sub_29DD190(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r14
  unsigned __int64 v7; // r12
  __int64 v8; // rbx
  unsigned __int8 *v9; // r15
  int v10; // eax
  __int64 v11; // rdx
  unsigned __int16 v12; // ax
  unsigned int v13; // eax
  __int64 v15; // rax
  __int64 v16; // rax
  unsigned int v17; // r15d
  bool v18; // al
  __int64 v19; // rax
  unsigned int v20; // eax
  bool v21; // zf
  unsigned __int8 *v22; // rax
  _BYTE *v23; // rcx
  char v24; // al
  bool v25; // al
  int v26; // eax
  unsigned __int8 **v27; // rax
  __int64 v28; // rax
  _BYTE *v29; // rax
  unsigned __int8 v30; // al
  int v31; // edx
  __int64 v32; // rax
  int v33; // eax
  bool v34; // al
  __int64 v35; // rax
  __int64 v36; // rax
  unsigned __int8 *v37; // [rsp+0h] [rbp-40h]
  _BYTE *v38; // [rsp+0h] [rbp-40h]
  unsigned int v39; // [rsp+0h] [rbp-40h]
  __int64 v40; // [rsp+8h] [rbp-38h]
  __int64 v41; // [rsp+8h] [rbp-38h]
  __int64 v42; // [rsp+8h] [rbp-38h]

  v6 = a3;
  v7 = a2;
  if ( *(_BYTE *)a1 == 3 && (*(_BYTE *)(a1 + 80) & 2) != 0 )
    *(_DWORD *)(a2 + 8) = 2;
  v8 = *(_QWORD *)(a1 + 16);
  if ( !v8 )
    return 0;
  while ( 1 )
  {
    while ( 1 )
    {
      v9 = *(unsigned __int8 **)(v8 + 24);
      v10 = *v9;
      if ( (unsigned __int8)v10 <= 0x15u )
      {
        if ( (_BYTE)v10 != 5 || *(_BYTE *)(*((_QWORD *)v9 + 1) + 8LL) != 14 )
        {
          if ( !(unsigned __int8)sub_29DCFA0(*(_QWORD *)(v8 + 24), a2, a3, a4, a5, a6) )
            return 1;
          goto LABEL_9;
        }
        goto LABEL_40;
      }
      if ( (unsigned __int8)v10 <= 0x1Cu )
        return 1;
      if ( !*(_BYTE *)(v7 + 32) )
      {
        a4 = *(_QWORD *)(*((_QWORD *)v9 + 5) + 72LL);
        v11 = *(_QWORD *)(v7 + 24);
        if ( v11 )
        {
          if ( v11 != a4 )
          {
            *(_BYTE *)(v7 + 32) = 1;
            v10 = *v9;
          }
        }
        else
        {
          *(_QWORD *)(v7 + 24) = a4;
          v10 = *v9;
        }
      }
      if ( (_BYTE)v10 != 61 )
        break;
      *(_BYTE *)(v7 + 1) = 1;
      v12 = *((_WORD *)v9 + 1);
      if ( (v12 & 1) != 0 )
        return 1;
      a3 = *(unsigned int *)(v7 + 36);
      v13 = (v12 >> 7) & 7;
      if ( (_DWORD)a3 != 4 || (a4 = 6, v13 != 5) )
      {
        if ( v13 != 4 || (a4 = 6, (_DWORD)a3 != 5) )
        {
          if ( (unsigned int)a3 >= v13 )
            v13 = *(_DWORD *)(v7 + 36);
          a4 = v13;
        }
      }
      *(_DWORD *)(v7 + 36) = a4;
      v8 = *(_QWORD *)(v8 + 8);
      if ( !v8 )
        return 0;
    }
    if ( (_BYTE)v10 == 62 )
      break;
    if ( (((_BYTE)v10 - 63) & 0xEF) == 0 )
      goto LABEL_40;
    a3 = v10 & 0xFFFFFFFD;
    if ( (v10 & 0xFD) == 0x54 )
    {
      if ( !*(_BYTE *)(v6 + 28) )
        goto LABEL_79;
      v27 = *(unsigned __int8 ***)(v6 + 8);
      a4 = *(unsigned int *)(v6 + 20);
      a3 = (__int64)&v27[a4];
      if ( v27 != (unsigned __int8 **)a3 )
      {
        while ( v9 != *v27 )
        {
          if ( (unsigned __int8 **)a3 == ++v27 )
            goto LABEL_70;
        }
        goto LABEL_9;
      }
LABEL_70:
      if ( (unsigned int)a4 < *(_DWORD *)(v6 + 16) )
      {
        *(_DWORD *)(v6 + 20) = a4 + 1;
        *(_QWORD *)a3 = v9;
        ++*(_QWORD *)v6;
      }
      else
      {
LABEL_79:
        a2 = (unsigned __int64)v9;
        sub_C8CC70(v6, (__int64)v9, a3, a4, a5, a6);
        if ( !(_BYTE)a3 )
          goto LABEL_9;
      }
      goto LABEL_40;
    }
    a3 = (unsigned int)(v10 - 82);
    if ( (unsigned __int8)(v10 - 82) <= 1u )
    {
      *(_BYTE *)v7 = 1;
      goto LABEL_9;
    }
    if ( (_BYTE)v10 != 85 )
    {
      v30 = v10 - 34;
      if ( v30 > 0x33u )
        return 1;
      a2 = 0x8000000000041uLL >> v30;
      if ( ((0x8000000000041uLL >> v30) & 1) == 0 )
        return 1;
LABEL_84:
      if ( (unsigned int)sub_B49240((__int64)v9) != 353 )
      {
        a5 = (__int64)(v9 - 32);
        if ( (unsigned __int8 *)v8 != v9 - 32 )
          return 1;
LABEL_86:
        *(_BYTE *)(v7 + 1) = 1;
        goto LABEL_9;
      }
LABEL_40:
      a2 = v7;
      if ( (unsigned __int8)sub_29DD190(v9, v7, v6) )
        return 1;
      goto LABEL_9;
    }
    v15 = *((_QWORD *)v9 - 4);
    if ( !v15 )
      goto LABEL_84;
    if ( *(_BYTE *)v15
      || (a2 = *((_QWORD *)v9 + 10), *(_QWORD *)(v15 + 24) != a2)
      || (*(_BYTE *)(v15 + 33) & 0x20) == 0
      || (v31 = *(_DWORD *)(v15 + 36), v31 != 238) && (unsigned int)(v31 - 240) > 1 )
    {
      if ( !*(_BYTE *)v15 )
      {
        a2 = *((_QWORD *)v9 + 10);
        if ( *(_QWORD *)(v15 + 24) == a2
          && (*(_BYTE *)(v15 + 33) & 0x20) != 0
          && ((*(_DWORD *)(v15 + 36) - 243) & 0xFFFFFFFD) == 0 )
        {
          a3 = *((_DWORD *)v9 + 1) & 0x7FFFFFF;
          v16 = *(_QWORD *)&v9[32 * (3 - a3)];
          v17 = *(_DWORD *)(v16 + 32);
          if ( v17 <= 0x40 )
            v18 = *(_QWORD *)(v16 + 24) == 0;
          else
            v18 = v17 == (unsigned int)sub_C444A0(v16 + 24);
          if ( !v18 )
            return 1;
LABEL_61:
          *(_DWORD *)(v7 + 8) = 3;
          goto LABEL_9;
        }
      }
      goto LABEL_84;
    }
    a3 = *((_DWORD *)v9 + 1) & 0x7FFFFFF;
    v32 = *(_QWORD *)&v9[32 * (3 - a3)];
    a4 = *(unsigned int *)(v32 + 32);
    if ( (unsigned int)a4 <= 0x40 )
    {
      v34 = *(_QWORD *)(v32 + 24) == 0;
    }
    else
    {
      v39 = *(_DWORD *)(v32 + 32);
      v42 = *((_DWORD *)v9 + 1) & 0x7FFFFFF;
      v33 = sub_C444A0(v32 + 24);
      a4 = v39;
      a3 = v42;
      v34 = v39 == v33;
    }
    if ( !v34 )
      return 1;
    v35 = *(_QWORD *)&v9[-32 * a3];
    if ( a1 == v35 && v35 )
    {
      *(_DWORD *)(v7 + 8) = 3;
      a3 = *((_DWORD *)v9 + 1) & 0x7FFFFFF;
    }
    v36 = *(_QWORD *)&v9[32 * (1 - a3)];
    if ( a1 == v36 && v36 )
      goto LABEL_86;
LABEL_9:
    v8 = *(_QWORD *)(v8 + 8);
    if ( !v8 )
      return 0;
  }
  v19 = *((_QWORD *)v9 - 8);
  if ( (a1 != v19 || !v19) && (v9[2] & 1) == 0 )
  {
    ++*(_DWORD *)(v7 + 4);
    a3 = *(unsigned int *)(v7 + 36);
    v20 = (*((_WORD *)v9 + 1) >> 7) & 7;
    if ( (_DWORD)a3 != 4 || (a4 = 6, v20 != 5) )
    {
      if ( v20 != 4 || (a4 = 6, (_DWORD)a3 != 5) )
      {
        if ( (unsigned int)a3 >= v20 )
          v20 = *(_DWORD *)(v7 + 36);
        a4 = v20;
      }
    }
    v21 = *(_DWORD *)(v7 + 8) == 3;
    *(_DWORD *)(v7 + 36) = a4;
    if ( v21 )
      goto LABEL_9;
    v22 = sub_BD3990(*((unsigned __int8 **)v9 - 4), a2);
    a3 = (__int64)v22;
    if ( *v22 != 3 )
      goto LABEL_61;
    v23 = (_BYTE *)*((_QWORD *)v9 - 8);
    if ( *v23 <= 0x15u )
    {
      v40 = *((_QWORD *)v9 - 8);
      v37 = v22;
      v24 = sub_AC2D00(v40);
      v23 = (_BYTE *)v40;
      a3 = (__int64)v37;
      if ( v24 )
        return 1;
    }
    v38 = v23;
    v41 = a3;
    v25 = sub_B2FC80(a3);
    a3 = v41;
    a4 = (__int64)v38;
    if ( v25 || v38 != *(_BYTE **)(v41 - 32) )
    {
      v26 = *(_DWORD *)(v7 + 8);
      if ( *v38 == 61 )
      {
        a2 = *((_QWORD *)v38 - 4);
        if ( v41 == a2 )
        {
          if ( a2 )
          {
            if ( v26 > 0 )
              goto LABEL_9;
LABEL_77:
            *(_DWORD *)(v7 + 8) = 1;
            goto LABEL_9;
          }
        }
      }
      if ( v26 <= 1 )
      {
        *(_DWORD *)(v7 + 8) = 2;
        *(_QWORD *)(v7 + 16) = v9;
        goto LABEL_9;
      }
      if ( v26 == 2 )
      {
        v28 = *(_QWORD *)(v7 + 16);
        if ( v28 )
        {
          v29 = *(_BYTE **)(v28 - 64);
          if ( v29 )
          {
            if ( v38 == v29 )
              goto LABEL_9;
          }
        }
      }
      goto LABEL_61;
    }
    if ( *(int *)(v7 + 8) <= 0 )
      goto LABEL_77;
    goto LABEL_9;
  }
  return 1;
}
