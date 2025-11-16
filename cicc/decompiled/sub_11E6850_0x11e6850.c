// Function: sub_11E6850
// Address: 0x11e6850
//
__int64 __fastcall sub_11E6850(__int64 a1, unsigned __int8 *a2, __int64 a3, int a4)
{
  __int64 v5; // r12
  char v7; // al
  int v8; // edx
  __int64 v9; // r12
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // r15
  int v13; // r15d
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rcx
  _BYTE *v17; // rax
  _BYTE *v18; // r12
  const char *v19; // rdi
  __int64 v20; // rdx
  __int64 *v21; // rax

  v5 = *((_QWORD *)a2 - 4);
  if ( v5 )
  {
    if ( *(_BYTE *)v5 )
    {
      v5 = 0;
    }
    else if ( *(_QWORD *)(v5 + 24) != *((_QWORD *)a2 + 10) )
    {
      v5 = 0;
    }
  }
  if ( (unsigned __int8)sub_A73ED0((_QWORD *)a2 + 9, 5) )
    return 0;
  v7 = sub_B49560((__int64)a2, 5);
  if ( !v5 || v7 || !sub_B2FC80(v5) )
    return 0;
  if ( a4 < 0 )
  {
LABEL_30:
    v21 = (__int64 *)sub_BD5C60((__int64)a2);
    *((_QWORD *)a2 + 9) = sub_A7A090((__int64 *)a2 + 9, v21, -1, 5);
    return 0;
  }
  v8 = *a2;
  if ( v8 == 40 )
  {
    v9 = 32LL * (unsigned int)sub_B491D0((__int64)a2);
  }
  else
  {
    v9 = 0;
    if ( v8 != 85 )
    {
      v9 = 64;
      if ( v8 != 34 )
        BUG();
    }
  }
  if ( (a2[7] & 0x80u) == 0 )
    goto LABEL_23;
  v10 = sub_BD2BC0((__int64)a2);
  v12 = v10 + v11;
  if ( (a2[7] & 0x80u) == 0 )
  {
    if ( (unsigned int)(v12 >> 4) )
LABEL_34:
      BUG();
LABEL_23:
    v16 = 0;
    goto LABEL_24;
  }
  if ( !(unsigned int)((v12 - sub_BD2BC0((__int64)a2)) >> 4) )
    goto LABEL_23;
  if ( (a2[7] & 0x80u) == 0 )
    goto LABEL_34;
  v13 = *(_DWORD *)(sub_BD2BC0((__int64)a2) + 8);
  if ( (a2[7] & 0x80u) == 0 )
    BUG();
  v14 = sub_BD2BC0((__int64)a2);
  v16 = 32LL * (unsigned int)(*(_DWORD *)(v14 + v15 - 4) - v13);
LABEL_24:
  if ( a4 < (int)((32LL * (*((_DWORD *)a2 + 1) & 0x7FFFFFF) - 32 - v9 - v16) >> 5) )
  {
    v17 = *(_BYTE **)&a2[32 * ((unsigned int)a4 - (unsigned __int64)(*((_DWORD *)a2 + 1) & 0x7FFFFFF))];
    if ( *v17 == 61 )
    {
      v18 = (_BYTE *)*((_QWORD *)v17 - 4);
      if ( *v18 == 3 && sub_B2FC80(*((_QWORD *)v17 - 4)) )
      {
        v19 = sub_BD5D20((__int64)v18);
        if ( v20 == 6 && !memcmp(v19, "stderr", 6u) )
          goto LABEL_30;
      }
    }
  }
  return 0;
}
