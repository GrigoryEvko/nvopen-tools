// Function: sub_26E0290
// Address: 0x26e0290
//
__int64 *__fastcall sub_26E0290(__int64 *a1, __int64 a2)
{
  __int64 v3; // rbx
  bool v4; // al
  __int64 v5; // rdx
  __int64 v6; // rdi
  __int64 v7; // r13
  unsigned __int8 v8; // al
  unsigned __int8 **v9; // rbx
  unsigned __int8 *v10; // rax
  __int64 v11; // rdx
  unsigned __int8 *v12; // rbx
  const char *v13; // rax
  unsigned __int8 v14; // al
  unsigned __int8 *v15; // r14
  __int64 v16; // rdi
  __int64 v18; // rcx
  unsigned __int8 *v19; // r14

  v3 = a2;
  v4 = (*(_BYTE *)(a2 - 16) & 2) != 0;
  while ( 1 )
  {
    if ( v4 )
    {
      if ( *(_DWORD *)(v3 - 24) != 2 )
        goto LABEL_29;
      v5 = *(_QWORD *)(v3 - 32);
    }
    else
    {
      if ( ((*(_WORD *)(v3 - 16) >> 6) & 0xF) != 2 )
LABEL_29:
        BUG();
      v5 = v3 - 16 - 8LL * ((*(_BYTE *)(v3 - 16) >> 2) & 0xF);
    }
    v6 = *(_QWORD *)(v5 + 8);
    v4 = (*(_BYTE *)(v6 - 16) & 2) != 0;
    if ( (*(_BYTE *)(v6 - 16) & 2) != 0 )
    {
      if ( *(_DWORD *)(v6 - 24) != 2 )
        break;
      v18 = *(_QWORD *)(v6 - 32);
    }
    else
    {
      if ( ((*(_WORD *)(v6 - 16) >> 6) & 0xF) != 2 )
        break;
      v18 = v6 - 16 - 8LL * ((*(_BYTE *)(v6 - 16) >> 2) & 0xF);
    }
    if ( !*(_QWORD *)(v18 + 8) )
      break;
    v3 = *(_QWORD *)(v5 + 8);
  }
  v7 = sub_C1B090(v6, unk_4F838D0);
  v8 = *(_BYTE *)(v3 - 16);
  if ( (v8 & 2) != 0 )
    v9 = *(unsigned __int8 ***)(v3 - 32);
  else
    v9 = (unsigned __int8 **)(v3 - 16 - 8LL * ((v8 >> 2) & 0xF));
  v10 = sub_AF34D0(*v9);
  v11 = 0;
  v12 = v10;
  v13 = byte_3F871B3;
  if ( v12 )
  {
    v14 = *(v12 - 16);
    v15 = v12 - 16;
    if ( (v14 & 2) != 0 )
    {
      v16 = *(_QWORD *)(*((_QWORD *)v12 - 4) + 24LL);
      if ( !v16 )
      {
LABEL_27:
        v19 = (unsigned __int8 *)*((_QWORD *)v12 - 4);
LABEL_24:
        v13 = (const char *)*((_QWORD *)v19 + 2);
        if ( v13 )
          v13 = (const char *)sub_B91420(*((_QWORD *)v19 + 2));
        else
          v11 = 0;
        goto LABEL_13;
      }
    }
    else
    {
      v16 = *(_QWORD *)&v15[-8 * ((v14 >> 2) & 0xF) + 24];
      if ( !v16 )
        goto LABEL_23;
    }
    v13 = (const char *)sub_B91420(v16);
    if ( v11 )
      goto LABEL_13;
    v14 = *(v12 - 16);
    if ( (v14 & 2) != 0 )
      goto LABEL_27;
LABEL_23:
    v19 = &v15[-8 * ((v14 >> 2) & 0xF)];
    goto LABEL_24;
  }
LABEL_13:
  *a1 = v7;
  a1[1] = (__int64)v13;
  a1[2] = v11;
  return a1;
}
