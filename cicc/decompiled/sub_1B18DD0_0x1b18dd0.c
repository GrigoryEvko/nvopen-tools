// Function: sub_1B18DD0
// Address: 0x1b18dd0
//
__int64 __fastcall sub_1B18DD0(__int64 a1, __int64 a2, __int64 a3)
{
  char v5; // al
  __int64 v6; // rdi
  _QWORD *v8; // rax
  int v9; // edx
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // r13
  __int64 v15; // rcx
  __int64 v16; // r14
  int v17; // eax
  int v18; // eax
  __int64 v19; // [rsp+0h] [rbp-50h] BYREF
  char v20; // [rsp+8h] [rbp-48h] BYREF
  __int64 *v21; // [rsp+10h] [rbp-40h] BYREF
  char *v22; // [rsp+18h] [rbp-38h]

  v5 = *(_BYTE *)(a2 + 16);
  if ( (unsigned __int8)(v5 - 75) > 1u )
  {
    if ( v5 != 79 )
      goto LABEL_4;
    v10 = (*(_BYTE *)(a2 + 23) & 0x40) != 0 ? *(_QWORD *)(a2 - 8) : a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
    if ( (unsigned __int8)(*(_BYTE *)(*(_QWORD *)v10 + 16LL) - 75) > 1u )
      goto LABEL_4;
    v11 = *(_QWORD *)(*(_QWORD *)v10 + 8LL);
    if ( !v11 || *(_QWORD *)(v11 + 8) )
      goto LABEL_4;
    v12 = *(_QWORD *)(a2 - 72);
    if ( *(_BYTE *)(v12 + 16) != 75 )
      goto LABEL_24;
    v13 = *(_QWORD *)(a2 - 48);
    v14 = *(_QWORD *)(v12 - 48);
    v15 = *(_QWORD *)(a2 - 24);
    v16 = *(_QWORD *)(v12 - 24);
    if ( v13 == v14 && v15 == v16 )
    {
      v17 = *(unsigned __int16 *)(v12 + 18);
    }
    else
    {
      if ( v13 != v16 || v15 != v14 )
      {
LABEL_40:
        if ( v13 != v16 || v14 != v15 )
          goto LABEL_24;
        if ( v13 != v14 )
        {
          v18 = sub_15FF0F0(*(_WORD *)(v12 + 18) & 0x7FFF);
          goto LABEL_30;
        }
LABEL_29:
        v18 = *(unsigned __int16 *)(v12 + 18);
        BYTE1(v18) &= ~0x80u;
LABEL_30:
        if ( (unsigned int)(v18 - 34) <= 1 )
        {
          if ( v14 )
          {
            v19 = v14;
            if ( v16 )
            {
              *(_BYTE *)a1 = 1;
              *(_QWORD *)(a1 + 8) = a2;
              *(_DWORD *)(a1 + 16) = 2;
              *(_QWORD *)(a1 + 24) = 0;
              return a1;
            }
          }
        }
        goto LABEL_24;
      }
      v17 = *(unsigned __int16 *)(v12 + 18);
      if ( v13 != v14 )
      {
        v17 = sub_15FF0F0(v17 & 0xFFFF7FFF);
        goto LABEL_19;
      }
    }
    BYTE1(v17) &= ~0x80u;
LABEL_19:
    if ( (unsigned int)(v17 - 36) <= 1 )
    {
      if ( v14 )
      {
        v19 = v14;
        if ( v16 )
        {
          *(_BYTE *)a1 = 1;
          *(_QWORD *)(a1 + 8) = a2;
          *(_DWORD *)(a1 + 16) = 1;
          *(_QWORD *)(a1 + 24) = 0;
          return a1;
        }
      }
    }
    v12 = *(_QWORD *)(a2 - 72);
    if ( *(_BYTE *)(v12 + 16) == 75 )
    {
      v13 = *(_QWORD *)(a2 - 48);
      v14 = *(_QWORD *)(v12 - 48);
      v15 = *(_QWORD *)(a2 - 24);
      v16 = *(_QWORD *)(v12 - 24);
      if ( v13 == v14 && v15 == v16 )
        goto LABEL_29;
      goto LABEL_40;
    }
LABEL_24:
    v21 = &v19;
    v22 = &v20;
    if ( sub_1B189B0(&v21, a2) )
    {
      *(_BYTE *)a1 = 1;
      *(_QWORD *)(a1 + 8) = a2;
      *(_DWORD *)(a1 + 16) = 4;
      *(_QWORD *)(a1 + 24) = 0;
      return a1;
    }
    v21 = &v19;
    v22 = &v20;
    if ( sub_1B18A60(&v21, a2) )
    {
      *(_BYTE *)a1 = 1;
      *(_QWORD *)(a1 + 8) = a2;
      *(_DWORD *)(a1 + 16) = 3;
      *(_QWORD *)(a1 + 24) = 0;
      return a1;
    }
    v21 = &v19;
    v22 = &v20;
    if ( sub_1B18B10(&v21, a2) )
    {
LABEL_45:
      *(_BYTE *)a1 = 1;
      *(_QWORD *)(a1 + 8) = a2;
      *(_DWORD *)(a1 + 16) = 5;
      *(_QWORD *)(a1 + 24) = 0;
      return a1;
    }
    v21 = &v19;
    v22 = &v20;
    if ( !sub_1B18BC0(&v21, a2) )
    {
      v21 = &v19;
      v22 = &v20;
      if ( sub_1B18C70(&v21, a2) )
        goto LABEL_45;
      v21 = &v19;
      v22 = &v20;
      if ( !sub_1B18D20(&v21, a2) )
        goto LABEL_4;
    }
    *(_BYTE *)a1 = 1;
    *(_QWORD *)(a1 + 8) = a2;
    *(_DWORD *)(a1 + 16) = 6;
    *(_QWORD *)(a1 + 24) = 0;
    return a1;
  }
  v6 = *(_QWORD *)(a2 + 8);
  if ( v6 )
  {
    if ( !*(_QWORD *)(v6 + 8) )
    {
      v8 = sub_1648700(v6);
      if ( *((_BYTE *)v8 + 16) == 79 )
      {
        v9 = *(_DWORD *)(a3 + 16);
        *(_BYTE *)a1 = 1;
        *(_QWORD *)(a1 + 8) = v8;
        *(_DWORD *)(a1 + 16) = v9;
        *(_QWORD *)(a1 + 24) = 0;
        return a1;
      }
    }
  }
LABEL_4:
  *(_BYTE *)a1 = 0;
  *(_QWORD *)(a1 + 8) = a2;
  *(_DWORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 24) = 0;
  return a1;
}
