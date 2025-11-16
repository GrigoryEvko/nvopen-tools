// Function: sub_AD8850
// Address: 0xad8850
//
__int64 __fastcall sub_AD8850(unsigned __int64 a1, unsigned __int64 a2)
{
  __int64 v2; // rbx
  unsigned int v4; // r13d
  __int64 v6; // rdi
  unsigned __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rax
  __int64 v12; // rdi
  int v13; // eax
  __int64 **v14; // r13
  unsigned __int8 *v15; // r14
  unsigned __int8 *v16; // rax
  char *v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rbx
  char v20; // al
  unsigned int v21; // r12d
  __int64 v22; // r12
  _BYTE *v23; // rax
  unsigned int v24; // ebx
  int v25; // r12d
  unsigned int v26; // r14d
  __int64 v27; // rax
  unsigned int v28; // r13d
  _QWORD v29[6]; // [rsp+0h] [rbp-30h] BYREF

  if ( a1 == a2 )
    return 1;
  v2 = *(_QWORD *)(a1 + 8);
  if ( (unsigned int)*(unsigned __int8 *)(v2 + 8) - 17 <= 1 )
  {
    v4 = 0;
    if ( *(_BYTE *)a2 > 0x15u || *(_QWORD *)(a2 + 8) != v2 )
      return v4;
    v6 = *(_QWORD *)(v2 + 24);
    v7 = *(unsigned __int8 *)(v6 + 8);
    if ( (unsigned __int8)v7 > 0xCu || (v8 = 4143, !_bittest64(&v8, v7)) )
    {
      v4 = 0;
      if ( (v7 & 0xFD) != 4 )
        return v4;
    }
    v9 = sub_BCAE30(v6);
    v29[1] = v10;
    v29[0] = v9;
    v11 = sub_CA1930(v29);
    v12 = sub_BCCE00(*(_QWORD *)v2, v11);
    v13 = *(_DWORD *)(v2 + 32);
    BYTE4(v29[0]) = *(_BYTE *)(v2 + 8) == 18;
    LODWORD(v29[0]) = v13;
    v14 = (__int64 **)sub_BCE1B0(v12, v29[0]);
    v15 = (unsigned __int8 *)sub_AD4C90(a1, v14, 0);
    v16 = (unsigned __int8 *)sub_AD4C90(a2, v14, 0);
    v17 = (char *)sub_AAB310(0x20u, v15, v16);
    v19 = (__int64)v17;
    if ( v17 )
    {
      v20 = *v17;
      v4 = 1;
      if ( v20 == 13 )
        return v4;
      if ( v20 == 17 )
      {
        v21 = *(_DWORD *)(v19 + 32);
        if ( v21 <= 0x40 )
          LOBYTE(v4) = *(_QWORD *)(v19 + 24) == 1;
        else
          LOBYTE(v4) = v21 - 1 == (unsigned int)sub_C444A0(v19 + 24);
        return v4;
      }
      v22 = *(_QWORD *)(v19 + 8);
      if ( (unsigned int)*(unsigned __int8 *)(v22 + 8) - 17 <= 1 )
      {
        v23 = sub_AD7630(v19, 0, v18);
        if ( v23 && *v23 == 17 )
        {
          v24 = *((_DWORD *)v23 + 8);
          if ( v24 <= 0x40 )
            LOBYTE(v4) = *((_QWORD *)v23 + 3) == 1;
          else
            LOBYTE(v4) = v24 - 1 == (unsigned int)sub_C444A0(v23 + 24);
          return v4;
        }
        if ( *(_BYTE *)(v22 + 8) == 17 )
        {
          v25 = *(_DWORD *)(v22 + 32);
          if ( v25 )
          {
            v4 = 0;
            v26 = 0;
            while ( 1 )
            {
              v27 = sub_AD69F0((unsigned __int8 *)v19, v26);
              if ( !v27 )
                break;
              if ( *(_BYTE *)v27 != 13 )
              {
                if ( *(_BYTE *)v27 != 17 )
                  return 0;
                v28 = *(_DWORD *)(v27 + 32);
                if ( v28 <= 0x40 )
                {
                  if ( *(_QWORD *)(v27 + 24) != 1 )
                    return 0;
                }
                else if ( (unsigned int)sub_C444A0(v27 + 24) != v28 - 1 )
                {
                  return 0;
                }
                v4 = 1;
              }
              if ( v25 == ++v26 )
                return v4;
            }
          }
        }
      }
    }
  }
  return 0;
}
