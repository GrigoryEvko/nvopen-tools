// Function: sub_1794FA0
// Address: 0x1794fa0
//
__int64 __fastcall sub_1794FA0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  _BYTE *v4; // r12
  char v5; // al
  _BYTE *v7; // r13
  unsigned __int8 v8; // r14
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // r13
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // rcx
  __int64 v15; // rdx
  _BYTE *v16; // r12
  _BYTE *v17; // r13
  __int64 v18; // rax
  __int64 v19; // rax
  __int64 v20; // rax

  v5 = *(_BYTE *)(a2 + 16);
  if ( v5 == 51 )
  {
    if ( *(_QWORD *)a1 == *(_QWORD *)(a2 - 48) )
    {
      v7 = *(_BYTE **)(a2 - 24);
      v8 = v7[16];
      if ( v8 == 13 )
      {
        v4 = v7 + 24;
        if ( *((_DWORD *)v7 + 8) > 0x40u )
        {
          if ( (unsigned int)sub_16A5940((__int64)(v7 + 24)) == 1 )
            goto LABEL_25;
        }
        else
        {
          v9 = *((_QWORD *)v7 + 3);
          if ( v9 )
          {
            a3 = v9 - 1;
            if ( (v9 & (v9 - 1)) == 0 )
              goto LABEL_25;
          }
        }
      }
      LOBYTE(v4) = v8 <= 0x10u && *(_BYTE *)(*(_QWORD *)v7 + 8LL) == 16;
      if ( !(_BYTE)v4 )
        goto LABEL_4;
      v10 = sub_15A1020(v7, a2, a3, a4);
      if ( !v10 || *(_BYTE *)(v10 + 16) != 13 )
        goto LABEL_4;
      v11 = v10 + 24;
      if ( *(_DWORD *)(v10 + 32) > 0x40u )
      {
        if ( (unsigned int)sub_16A5940(v10 + 24) != 1 )
          goto LABEL_4;
      }
      else
      {
        v12 = *(_QWORD *)(v10 + 24);
        if ( !v12 || (v12 & (v12 - 1)) != 0 )
          goto LABEL_4;
      }
      **(_QWORD **)(a1 + 8) = v11;
      return (unsigned int)v4;
    }
  }
  else
  {
    if ( v5 != 5 )
      goto LABEL_4;
    if ( *(_WORD *)(a2 + 18) != 27 )
      goto LABEL_4;
    v13 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
    v14 = *(_QWORD *)(a2 - 24 * v13);
    if ( *(_QWORD *)a1 != v14 )
      goto LABEL_4;
    v15 = 1 - v13;
    v16 = *(_BYTE **)(a2 + 24 * (1 - v13));
    if ( v16[16] == 13 )
    {
      v17 = v16 + 24;
      if ( *((_DWORD *)v16 + 8) > 0x40u )
      {
        if ( (unsigned int)sub_16A5940((__int64)(v16 + 24)) == 1 )
        {
LABEL_23:
          LODWORD(v4) = 1;
          **(_QWORD **)(a1 + 8) = v17;
          return (unsigned int)v4;
        }
      }
      else
      {
        v18 = *((_QWORD *)v16 + 3);
        if ( v18 )
        {
          v15 = v18 - 1;
          if ( (v18 & (v18 - 1)) == 0 )
            goto LABEL_23;
        }
      }
    }
    if ( *(_BYTE *)(*(_QWORD *)v16 + 8LL) != 16 )
      goto LABEL_4;
    v19 = sub_15A1020(v16, a2, v15, v14);
    if ( !v19 || *(_BYTE *)(v19 + 16) != 13 )
      goto LABEL_4;
    v4 = (_BYTE *)(v19 + 24);
    if ( *(_DWORD *)(v19 + 32) <= 0x40u )
    {
      v20 = *(_QWORD *)(v19 + 24);
      if ( !v20 || (v20 & (v20 - 1)) != 0 )
        goto LABEL_4;
LABEL_25:
      **(_QWORD **)(a1 + 8) = v4;
      LODWORD(v4) = 1;
      return (unsigned int)v4;
    }
    if ( (unsigned int)sub_16A5940(v19 + 24) == 1 )
      goto LABEL_25;
  }
LABEL_4:
  LODWORD(v4) = 0;
  return (unsigned int)v4;
}
