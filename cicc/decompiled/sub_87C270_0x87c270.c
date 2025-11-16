// Function: sub_87C270
// Address: 0x87c270
//
__int64 __fastcall sub_87C270(__int64 a1, __int64 a2, _DWORD *a3)
{
  __int64 i; // r14
  unsigned int v5; // eax
  int v6; // r15d
  __int64 v7; // r8
  char v8; // al
  __int64 v9; // rdx
  __int64 v10; // rax
  _BOOL8 v11; // rax
  __int64 v12; // r12
  __int64 v13; // rbx
  int v14; // edx
  int v15; // r15d
  __int64 v17; // rax
  bool v18; // [rsp+Eh] [rbp-92h]
  bool v19; // [rsp+Fh] [rbp-91h]
  int v20; // [rsp+14h] [rbp-8Ch] BYREF
  int v21; // [rsp+18h] [rbp-88h] BYREF
  int v22; // [rsp+1Ch] [rbp-84h] BYREF
  __int64 v23; // [rsp+20h] [rbp-80h]
  __int64 v24; // [rsp+28h] [rbp-78h]
  _BYTE v25[32]; // [rsp+30h] [rbp-70h] BYREF
  _QWORD v26[2]; // [rsp+50h] [rbp-50h]
  _QWORD v27[8]; // [rsp+60h] [rbp-40h]

  i = a1;
  v19 = 0;
  v18 = (*(_BYTE *)(a1 + 81) & 0x10) != 0;
  if ( dword_4D04818 )
  {
    if ( *(char *)(a2 + 142) >= 0 && *(_BYTE *)(a2 + 140) == 12 )
      v5 = sub_8D4AB0(a2, a2, a3);
    else
      v5 = *(_DWORD *)(a2 + 136);
    v19 = dword_4F06980 < v5;
  }
  v6 = 0;
LABEL_7:
  *a3 = 0;
  v26[0] = 0;
  v26[1] = 0;
  v27[0] = 0;
  v27[1] = 0;
  v23 = 0;
  v24 = 0;
  for ( i = sub_82C1B0(i, 0, 0, (__int64)v25); i; i = sub_82C230(v25) )
  {
    while ( 1 )
    {
      v8 = *(_BYTE *)(i + 80);
      v9 = i;
      if ( v8 == 16 )
      {
        if ( (*(_BYTE *)(i + 82) & 4) != 0 )
        {
          *a3 = 1;
          return 0;
        }
        v9 = **(_QWORD **)(i + 88);
        v8 = *(_BYTE *)(v9 + 80);
        if ( v8 == 24 )
          break;
      }
      if ( (unsigned __int8)(v8 - 10) <= 1u )
        goto LABEL_16;
LABEL_10:
      if ( v8 == 17 )
        goto LABEL_16;
LABEL_11:
      i = sub_82C230(v25);
      if ( !i )
        goto LABEL_22;
    }
    v9 = *(_QWORD *)(v9 + 88);
    v8 = *(_BYTE *)(v9 + 80);
    if ( (unsigned __int8)(v8 - 10) > 1u )
      goto LABEL_10;
LABEL_16:
    if ( !sub_87ADD0(*(_QWORD *)(v9 + 88), &v20, &v21, &v22, v7) )
      goto LABEL_11;
    if ( unk_4D04814 )
    {
      if ( v22 )
      {
        if ( !v6 )
        {
          v6 = 1;
          goto LABEL_7;
        }
      }
      else if ( v6 )
      {
        goto LABEL_11;
      }
    }
    v10 = v21 + 2LL * v20;
    if ( !v26[v10] )
    {
      v26[v10] = i;
      goto LABEL_11;
    }
    *((_DWORD *)&v23 + v10) = 1;
  }
LABEL_22:
  if ( *a3 )
    return 0;
  v11 = v19;
  v12 = v26[v11];
  v13 = v27[v11];
  if ( v12 || v13 || !dword_4D04818 )
  {
    v14 = *((_DWORD *)&v23 + v11);
    v15 = *(_DWORD *)&v25[4 * v11 - 8];
    *a3 = v14;
  }
  else
  {
    v17 = 1 - v19;
    v14 = *((_DWORD *)&v23 + v17);
    v12 = v26[v17];
    v17 += 2;
    v13 = v26[v17];
    v15 = *((_DWORD *)&v23 + v17);
    *a3 = v14;
  }
  if ( v14 )
    return 0;
  if ( !v13 )
  {
    if ( v12 )
      return v12;
    goto LABEL_41;
  }
  if ( !v12 )
  {
LABEL_41:
    if ( !v18 )
      return v12;
    goto LABEL_42;
  }
  if ( unk_4D04478
    && !v18
    && !(unsigned int)sub_8D23B0(a2)
    && (*(_BYTE *)(*(_QWORD *)(v13 + 88) + 176LL) != 4 || (unsigned int)sub_691630(a2, 0)) )
  {
LABEL_42:
    if ( v15 )
      *a3 = 1;
    else
      return v13;
  }
  return v12;
}
