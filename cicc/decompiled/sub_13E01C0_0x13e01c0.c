// Function: sub_13E01C0
// Address: 0x13e01c0
//
unsigned __int8 *__fastcall sub_13E01C0(unsigned __int8 *a1, unsigned __int8 *a2, _QWORD *a3, int a4)
{
  unsigned __int8 *v6; // r12
  unsigned __int8 v7; // al
  unsigned __int8 *v8; // r15
  unsigned __int8 v10; // al
  unsigned __int8 v11; // al
  __int64 v12; // rdi
  __int64 v13; // rax
  unsigned int v14; // ebx
  bool v15; // al
  unsigned __int8 v16; // al
  unsigned int v17; // ebx
  bool v18; // al
  __int64 v19; // rax
  unsigned int v20; // ebx
  unsigned int v21; // edx
  __int64 v22; // rax
  __int64 v23; // rax
  unsigned int v24; // ebx
  int v25; // eax
  unsigned int v26; // edx
  __int64 v27; // rdx
  unsigned __int8 *v28; // rax
  unsigned int v29; // ebx
  __int64 v30; // rax
  char v31; // cl
  bool v32; // al
  unsigned __int8 v33; // al
  unsigned int v34; // ebx
  __int64 v35; // rax
  char v36; // cl
  bool v37; // al
  int v38; // [rsp+8h] [rbp-58h]
  int v39; // [rsp+8h] [rbp-58h]
  int v40; // [rsp+Ch] [rbp-54h]
  int v41; // [rsp+Ch] [rbp-54h]
  __int64 v42; // [rsp+18h] [rbp-48h] BYREF
  _QWORD v43[8]; // [rsp+20h] [rbp-40h] BYREF

  v6 = a1;
  v7 = a1[16];
  if ( v7 > 0x10u )
  {
LABEL_5:
    v10 = a2[16];
    if ( v10 == 9 )
    {
      v8 = a1;
      return (unsigned __int8 *)sub_15A06D0(*(_QWORD *)v8);
    }
    v8 = a1;
    v6 = a2;
    if ( v10 > 0x10u )
      goto LABEL_7;
LABEL_17:
    if ( (unsigned __int8)sub_1593BB0(v6) )
      return (unsigned __int8 *)sub_15A06D0(*(_QWORD *)v8);
    if ( v6[16] == 13 )
    {
      v14 = *((_DWORD *)v6 + 8);
      if ( v14 <= 0x40 )
        v15 = *((_QWORD *)v6 + 3) == 0;
      else
        v15 = v14 == (unsigned int)sub_16A57B0(v6 + 24);
    }
    else
    {
      if ( *(_BYTE *)(*(_QWORD *)v6 + 8LL) != 16 )
        goto LABEL_7;
      v19 = sub_15A1020(v6);
      if ( !v19 || *(_BYTE *)(v19 + 16) != 13 )
      {
        v41 = *(_QWORD *)(*(_QWORD *)v6 + 32LL);
        if ( !v41 )
          return (unsigned __int8 *)sub_15A06D0(*(_QWORD *)v8);
        v34 = 0;
        while ( 1 )
        {
          v35 = sub_15A0A60(v6, v34);
          if ( !v35 )
            break;
          v36 = *(_BYTE *)(v35 + 16);
          if ( v36 != 9 )
          {
            if ( v36 != 13 )
              break;
            if ( *(_DWORD *)(v35 + 32) <= 0x40u )
            {
              v37 = *(_QWORD *)(v35 + 24) == 0;
            }
            else
            {
              v39 = *(_DWORD *)(v35 + 32);
              v37 = v39 == (unsigned int)sub_16A57B0(v35 + 24);
            }
            if ( !v37 )
              break;
          }
          if ( v41 == ++v34 )
            return (unsigned __int8 *)sub_15A06D0(*(_QWORD *)v8);
        }
        goto LABEL_24;
      }
      v20 = *(_DWORD *)(v19 + 32);
      if ( v20 <= 0x40 )
        v15 = *(_QWORD *)(v19 + 24) == 0;
      else
        v15 = v20 == (unsigned int)sub_16A57B0(v19 + 24);
    }
    if ( v15 )
      return (unsigned __int8 *)sub_15A06D0(*(_QWORD *)v8);
LABEL_24:
    v16 = v6[16];
    if ( v16 == 13 )
    {
      v17 = *((_DWORD *)v6 + 8);
      if ( v17 <= 0x40 )
        v18 = *((_QWORD *)v6 + 3) == 1;
      else
        v18 = v17 - 1 == (unsigned int)sub_16A57B0(v6 + 24);
    }
    else
    {
      if ( *(_BYTE *)(*(_QWORD *)v6 + 8LL) != 16 || v16 > 0x10u )
        goto LABEL_7;
      v23 = sub_15A1020(v6);
      if ( !v23 || *(_BYTE *)(v23 + 16) != 13 )
      {
        v40 = *(_QWORD *)(*(_QWORD *)v6 + 32LL);
        if ( !v40 )
          return v8;
        v29 = 0;
        while ( 1 )
        {
          v30 = sub_15A0A60(v6, v29);
          if ( !v30 )
            break;
          v31 = *(_BYTE *)(v30 + 16);
          if ( v31 != 9 )
          {
            if ( v31 != 13 )
              break;
            if ( *(_DWORD *)(v30 + 32) <= 0x40u )
            {
              v32 = *(_QWORD *)(v30 + 24) == 1;
            }
            else
            {
              v38 = *(_DWORD *)(v30 + 32);
              v32 = v38 - 1 == (unsigned int)sub_16A57B0(v30 + 24);
            }
            if ( !v32 )
              break;
          }
          if ( v40 == ++v29 )
            return v8;
        }
        goto LABEL_7;
      }
      v24 = *(_DWORD *)(v23 + 32);
      if ( v24 <= 0x40 )
        v18 = *(_QWORD *)(v23 + 24) == 1;
      else
        v18 = v24 - 1 == (unsigned int)sub_16A57B0(v23 + 24);
    }
    if ( v18 )
      return v8;
LABEL_7:
    v11 = v8[16];
    v42 = 0;
    if ( v11 > 0x17u )
    {
      v21 = v11 - 41;
      if ( ((unsigned __int8)(v11 - 48) <= 1u || v21 <= 1) && (v8[17] & 2) != 0 && v21 <= 1 )
      {
        v22 = *(_QWORD *)sub_13CF970((__int64)v8);
        if ( v22 )
        {
          v42 = v22;
          if ( v6 == *(unsigned __int8 **)(sub_13CF970((__int64)v8) + 24) )
            return (unsigned __int8 *)v42;
        }
      }
    }
    else if ( v11 == 5 )
    {
      v25 = *((unsigned __int16 *)v8 + 9);
      v26 = v25 - 17;
      if ( ((unsigned __int16)(v25 - 24) <= 1u || v26 <= 1) && (v8[17] & 2) != 0 && v26 <= 1 )
      {
        v27 = *((_DWORD *)v8 + 5) & 0xFFFFFFF;
        if ( *(_QWORD *)&v8[-24 * v27] )
        {
          v42 = *(_QWORD *)&v8[-24 * v27];
          v28 = *(unsigned __int8 **)&v8[24 * (1 - v27)];
          if ( v6 == v28 )
          {
            if ( v28 )
              return (unsigned __int8 *)v42;
          }
        }
      }
    }
    v43[1] = v8;
    v43[0] = &v42;
    if ( sub_13D24E0((__int64)v43, (__int64)v6) )
      return (unsigned __int8 *)v42;
    if ( !a4 )
      goto LABEL_94;
    v12 = *(_QWORD *)v8;
    if ( *(_BYTE *)(*(_QWORD *)v8 + 8LL) == 16 )
      v12 = **(_QWORD **)(v12 + 16);
    if ( !(unsigned __int8)sub_1642F90(v12, 1) || (v13 = sub_13DF820((__int64)v8, (__int64)v6, a3, a4 - 1)) == 0 )
    {
LABEL_94:
      v13 = (__int64)sub_13DDF20(15, v8, v6, a3, a4);
      if ( !v13 )
      {
        v13 = (__int64)sub_13DF2B0(15, v8, v6, 11, a3, a4);
        if ( !v13 )
        {
          v33 = v8[16];
          if ( v33 != 79 && v6[16] != 79 )
          {
LABEL_70:
            if ( v33 == 77 || v6[16] == 77 )
              return (unsigned __int8 *)sub_13DF6F0(15, v8, v6, a3, a4);
            else
              return 0;
          }
          v13 = (__int64)sub_13DF4D0(0xFu, v8, v6, a3, a4);
          if ( !v13 )
          {
            v33 = v8[16];
            goto LABEL_70;
          }
        }
      }
    }
    return (unsigned __int8 *)v13;
  }
  if ( a2[16] <= 0x10u )
  {
    v8 = (unsigned __int8 *)sub_14D6F90(15, a1, a2, *a3);
    if ( v8 )
      return v8;
    goto LABEL_5;
  }
  v8 = a2;
  if ( v7 != 9 )
    goto LABEL_17;
  return (unsigned __int8 *)sub_15A06D0(*(_QWORD *)v8);
}
