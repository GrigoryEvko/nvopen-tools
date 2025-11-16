// Function: sub_14BDDF0
// Address: 0x14bddf0
//
char __fastcall sub_14BDDF0(__int64 a1, unsigned __int8 a2, unsigned int a3, __int64 a4)
{
  unsigned __int8 v6; // r13
  unsigned int *v8; // rax
  unsigned int v9; // r8d
  unsigned int v10; // eax
  unsigned __int8 v11; // al
  __int64 v12; // rdi
  __int64 v14; // rax
  __int64 v15; // rax
  unsigned int v16; // edx
  __int64 v17; // rax
  char v18; // cl
  unsigned int v19; // edx
  int v20; // eax
  __int64 v21; // rax
  int v22; // eax
  int v23; // [rsp+8h] [rbp-48h]
  unsigned int v24; // [rsp+Ch] [rbp-44h]
  int v25; // [rsp+10h] [rbp-40h]
  __int64 v26; // [rsp+10h] [rbp-40h]
  int v27; // [rsp+10h] [rbp-40h]
  int v28; // [rsp+18h] [rbp-38h]
  unsigned int v29; // [rsp+18h] [rbp-38h]
  __int64 v30; // [rsp+18h] [rbp-38h]

  v6 = a2;
  v8 = (unsigned int *)sub_16D40F0(qword_4FBB370);
  if ( v8 )
    v10 = *v8;
  else
    v10 = qword_4FBB370[2];
  if ( a3 >= v10 )
    return 0;
  if ( !a2 )
    return sub_14BD070((_QWORD *)a1, a2, a3, a4, v9);
  v11 = *(_BYTE *)(a1 + 16);
  if ( v11 == 13 )
  {
    if ( *(_DWORD *)(a1 + 32) > 0x40u )
    {
      v25 = *(_DWORD *)(a1 + 32);
      v12 = a1 + 24;
      if ( v25 == (unsigned int)sub_16A57B0(a1 + 24) )
        return v6;
      goto LABEL_8;
    }
    v15 = *(_QWORD *)(a1 + 24);
    if ( !v15 )
      return v6;
LABEL_21:
    if ( (v15 & (v15 - 1)) != 0 )
      return sub_14BD070((_QWORD *)a1, a2, a3, a4, v9);
    return v6;
  }
  if ( *(_BYTE *)(*(_QWORD *)a1 + 8LL) != 16 || v11 > 0x10u )
    return sub_14BD070((_QWORD *)a1, a2, a3, a4, v9);
  v14 = sub_15A1020(a1);
  if ( !v14 || *(_BYTE *)(v14 + 16) != 13 )
  {
    v27 = *(_QWORD *)(*(_QWORD *)a1 + 32LL);
    if ( v27 )
    {
      v16 = 0;
      while ( 1 )
      {
        v29 = v16;
        v17 = sub_15A0A60(a1, v16);
        if ( !v17 )
          break;
        v18 = *(_BYTE *)(v17 + 16);
        v19 = v29;
        if ( v18 != 9 )
        {
          if ( v18 != 13 )
            return sub_14BD070((_QWORD *)a1, a2, a3, a4, v9);
          if ( *(_DWORD *)(v17 + 32) <= 0x40u )
          {
            v21 = *(_QWORD *)(v17 + 24);
            if ( v21 && (v21 & (v21 - 1)) != 0 )
              return sub_14BD070((_QWORD *)a1, a2, a3, a4, v9);
          }
          else
          {
            v23 = *(_DWORD *)(v17 + 32);
            v24 = v29;
            v30 = v17 + 24;
            v20 = sub_16A57B0(v17 + 24);
            v19 = v24;
            if ( v23 != v20 )
            {
              v22 = sub_16A5940(v30);
              v19 = v24;
              if ( v22 != 1 )
                return sub_14BD070((_QWORD *)a1, a2, a3, a4, v9);
            }
          }
        }
        v16 = v19 + 1;
        if ( v27 == v16 )
          return v6;
      }
      return sub_14BD070((_QWORD *)a1, a2, a3, a4, v9);
    }
    return v6;
  }
  if ( *(_DWORD *)(v14 + 32) <= 0x40u )
  {
    if ( !*(_QWORD *)(v14 + 24) )
      return v6;
    v15 = *(_QWORD *)(v14 + 24);
    if ( !v15 )
      return sub_14BD070((_QWORD *)a1, a2, a3, a4, v9);
    goto LABEL_21;
  }
  v28 = *(_DWORD *)(v14 + 32);
  v26 = v14 + 24;
  if ( v28 == (unsigned int)sub_16A57B0(v14 + 24) )
    return v6;
  v12 = v26;
LABEL_8:
  if ( (unsigned int)sub_16A5940(v12) != 1 )
    return sub_14BD070((_QWORD *)a1, a2, a3, a4, v9);
  return v6;
}
