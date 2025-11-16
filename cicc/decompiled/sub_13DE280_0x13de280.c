// Function: sub_13DE280
// Address: 0x13de280
//
unsigned __int8 *__fastcall sub_13DE280(unsigned __int8 *a1, unsigned __int8 *a2, _QWORD *a3, int a4)
{
  unsigned __int8 *v6; // r12
  unsigned __int8 v7; // al
  unsigned __int8 *v8; // r13
  unsigned __int8 v10; // al
  unsigned int v11; // ebx
  bool v12; // al
  __int64 v13; // rax
  unsigned int v14; // ebx
  unsigned int v15; // ebx
  __int64 v16; // rax
  char v17; // cl
  int v18; // [rsp+8h] [rbp-58h]
  int v19; // [rsp+Ch] [rbp-54h]
  unsigned __int8 *v20; // [rsp+10h] [rbp-50h] BYREF
  unsigned __int8 *v21; // [rsp+20h] [rbp-40h] BYREF

  v6 = a1;
  v7 = a1[16];
  if ( v7 > 0x10u )
    goto LABEL_5;
  if ( a2[16] <= 0x10u )
  {
    v8 = (unsigned __int8 *)sub_14D6F90(28, a1, a2, *a3);
    if ( v8 )
      return v8;
LABEL_5:
    v10 = a2[16];
    if ( v10 == 9 )
      return a2;
    v8 = a1;
    v6 = a2;
    if ( v10 > 0x10u )
      goto LABEL_7;
    goto LABEL_12;
  }
  v8 = a2;
  if ( v7 == 9 )
    return a1;
LABEL_12:
  if ( (unsigned __int8)sub_1593BB0(v6) )
    return v8;
  if ( v6[16] == 13 )
  {
    v11 = *((_DWORD *)v6 + 8);
    if ( v11 <= 0x40 )
      v12 = *((_QWORD *)v6 + 3) == 0;
    else
      v12 = v11 == (unsigned int)sub_16A57B0(v6 + 24);
  }
  else
  {
    if ( *(_BYTE *)(*(_QWORD *)v6 + 8LL) != 16 )
      goto LABEL_7;
    v13 = sub_15A1020(v6);
    if ( !v13 || *(_BYTE *)(v13 + 16) != 13 )
    {
      v19 = *(_QWORD *)(*(_QWORD *)v6 + 32LL);
      if ( !v19 )
        return v8;
      v15 = 0;
      while ( 1 )
      {
        v16 = sub_15A0A60(v6, v15);
        if ( !v16 )
          goto LABEL_7;
        v17 = *(_BYTE *)(v16 + 16);
        if ( v17 != 9 )
        {
          if ( v17 != 13 )
            goto LABEL_7;
          if ( *(_DWORD *)(v16 + 32) <= 0x40u )
          {
            if ( *(_QWORD *)(v16 + 24) )
              goto LABEL_7;
          }
          else
          {
            v18 = *(_DWORD *)(v16 + 32);
            if ( v18 != (unsigned int)sub_16A57B0(v16 + 24) )
              goto LABEL_7;
          }
        }
        if ( v19 == ++v15 )
          return v8;
      }
    }
    v14 = *(_DWORD *)(v13 + 32);
    if ( v14 <= 0x40 )
      v12 = *(_QWORD *)(v13 + 24) == 0;
    else
      v12 = v14 == (unsigned int)sub_16A57B0(v13 + 24);
  }
  if ( v12 )
    return v8;
LABEL_7:
  if ( v8 == v6 )
    return (unsigned __int8 *)sub_15A06D0(*(_QWORD *)v8);
  v20 = v6;
  if ( sub_13D1F50((__int64 *)&v20, (__int64)v8) )
    return (unsigned __int8 *)sub_15A04A0(*(_QWORD *)v8);
  v21 = v8;
  if ( sub_13D1F50((__int64 *)&v21, (__int64)v6) )
    return (unsigned __int8 *)sub_15A04A0(*(_QWORD *)v8);
  else
    return sub_13DDF20(28, v8, v6, a3, a4);
}
