// Function: sub_1B31CE0
// Address: 0x1b31ce0
//
__int64 __fastcall sub_1B31CE0(__int64 a1, char a2)
{
  __int64 v2; // r14
  __int64 v3; // rbx
  _QWORD *v4; // rax
  int v5; // r8d
  int v6; // r9d
  _QWORD *v7; // r15
  char v8; // al
  __int64 v9; // r13
  __int64 i; // r14
  _QWORD *v11; // rax
  unsigned __int8 v12; // dl
  _BYTE *v13; // r12
  char v14; // bl
  __int64 v15; // r13
  __int64 v16; // r15
  _QWORD *v17; // rax
  _QWORD *v18; // rdi
  __int64 v19; // rax
  __int64 v21; // rax
  __int64 v22; // rax
  _BYTE *v23; // [rsp+10h] [rbp-D0h]
  _QWORD *v24; // [rsp+10h] [rbp-D0h]
  _BYTE *v25; // [rsp+20h] [rbp-C0h] BYREF
  __int64 v26; // [rsp+28h] [rbp-B8h]
  _BYTE v27[176]; // [rsp+30h] [rbp-B0h] BYREF

  v2 = 0;
  v3 = *(_QWORD *)(a1 + 8);
  v25 = v27;
  v26 = 0x1000000000LL;
  if ( !v3 )
    return v2;
  do
  {
    while ( 1 )
    {
      v4 = sub_1648700(v3);
      v3 = *(_QWORD *)(v3 + 8);
      v7 = v4;
      v8 = *((_BYTE *)v4 + 16);
      if ( (unsigned __int8)(v8 - 54) > 1u )
        break;
LABEL_5:
      if ( !v3 )
        goto LABEL_19;
    }
    if ( !*(_BYTE *)(*v7 + 8LL) )
    {
      if ( !a2 || v8 == 78 && (v21 = *(v7 - 3), !*(_BYTE *)(v21 + 16)) && (*(_BYTE *)(v21 + 33) & 0x20) != 0 )
        sub_15F20C0(v7);
      goto LABEL_5;
    }
    v9 = v7[1];
    for ( i = (unsigned int)v26; v9; v9 = *(_QWORD *)(v9 + 8) )
    {
      v11 = sub_1648700(v9);
      v12 = *((_BYTE *)v11 + 16);
      if ( v12 > 0x17u && ((unsigned __int8)(v12 - 71) <= 1u || v12 == 56) )
      {
        if ( (unsigned int)i >= HIDWORD(v26) )
        {
          v24 = v11;
          sub_16CD150((__int64)&v25, v27, 0, 8, v5, v6);
          i = (unsigned int)v26;
          v11 = v24;
        }
        *(_QWORD *)&v25[8 * i] = v11;
        i = (unsigned int)(v26 + 1);
        LODWORD(v26) = v26 + 1;
      }
    }
    if ( HIDWORD(v26) <= (unsigned int)i )
    {
      sub_16CD150((__int64)&v25, v27, 0, 8, v5, v6);
      i = (unsigned int)v26;
    }
    *(_QWORD *)&v25[8 * i] = v7;
    LODWORD(v26) = v26 + 1;
  }
  while ( v3 );
LABEL_19:
  v13 = v25;
  v2 = 0;
  v23 = &v25[8 * (unsigned int)v26];
  if ( v25 != v23 )
  {
    v14 = 1;
    do
    {
      v15 = *(_QWORD *)v13;
      v16 = *(_QWORD *)(*(_QWORD *)v13 + 8LL);
      while ( v16 )
      {
        while ( 1 )
        {
          v17 = sub_1648700(v16);
          v16 = *(_QWORD *)(v16 + 8);
          v18 = v17;
          if ( *((_BYTE *)v17 + 16) == 78 )
          {
            v19 = *(v17 - 3);
            if ( !*(_BYTE *)(v19 + 16) && (*(_BYTE *)(v19 + 33) & 0x20) != 0 )
              break;
          }
          if ( !v16 )
            goto LABEL_30;
        }
        if ( v14 && *(_DWORD *)(v19 + 36) == 117 )
        {
          if ( v2 )
          {
            v2 = 0;
            v14 = 0;
          }
          else
          {
            v2 = v18[5];
          }
        }
        sub_15F20C0(v18);
      }
LABEL_30:
      if ( !a2
        || *(_BYTE *)(v15 + 16) == 78
        && (v22 = *(_QWORD *)(v15 - 24), !*(_BYTE *)(v22 + 16))
        && (*(_BYTE *)(v22 + 33) & 0x20) != 0 )
      {
        sub_15F20C0((_QWORD *)v15);
      }
      v13 += 8;
    }
    while ( v23 != v13 );
    v23 = v25;
  }
  if ( v23 != v27 )
    _libc_free((unsigned __int64)v23);
  return v2;
}
