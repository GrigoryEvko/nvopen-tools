// Function: sub_919DE0
// Address: 0x919de0
//
__int64 __fastcall sub_919DE0(_QWORD **a1, __int64 a2, unsigned __int8 a3, __int64 a4)
{
  __int64 v5; // rbx
  __int64 v6; // rax
  __int64 v7; // rax
  __int64 v8; // r14
  __int64 v9; // rax
  const char *v11; // r14
  __int64 v12; // rax
  bool v13; // zf
  __int64 v14; // r12
  size_t v15; // rax
  __int64 v16; // rcx
  unsigned int v17; // edx
  const char *v18; // r14
  size_t v19; // rdx
  __int64 v20; // rcx
  char v21; // r14
  __int64 v22; // rax
  __int64 v23; // rax
  _QWORD *v24; // rax
  __int64 v25; // rsi
  char v26; // al
  __int64 v27; // rax
  _QWORD *v28; // [rsp+0h] [rbp-40h] BYREF
  size_t v29; // [rsp+8h] [rbp-38h]
  _QWORD v30[6]; // [rsp+10h] [rbp-30h] BYREF

  v5 = a2;
  if ( *(_BYTE *)(a2 + 140) == 12 )
  {
    v6 = a2;
    do
      v6 = *(_QWORD *)(v6 + 160);
    while ( *(_BYTE *)(v6 + 140) == 12 );
    if ( a2 != v6 )
      sub_91B8A0("internal error while translating type!");
  }
  if ( (*(_BYTE *)(a2 + 89) & 1) != 0 )
  {
    v7 = sub_72B7D0(a2);
    v8 = v7;
    if ( v7 )
    {
      v9 = sub_72B840(v7);
      a2 = v8;
      sub_5E3920(v5, v8, *(_DWORD *)(v9 + 24));
      if ( *(_BYTE *)(v5 + 140) > 0xFu )
LABEL_10:
        sub_91B8A0("unsupported type during translation!");
    }
  }
  switch ( *(_BYTE *)(v5 + 140) )
  {
    case 1:
      v24 = *a1;
      v25 = 8;
      return sub_BCCE00(*v24, v25);
    case 2:
      v24 = *a1;
      v25 = 8 * (unsigned int)*(_QWORD *)(v5 + 128);
      return sub_BCCE00(*v24, v25);
    case 3:
      v26 = *(_BYTE *)(v5 + 160);
      if ( v26 == 2 )
        return sub_BCB160(**a1);
      if ( v26 != 4 )
      {
        if ( !dword_4D0470C && (v26 == 8 || v26 == 13) )
        {
          LOBYTE(a4) = v26 == 13;
          return sub_BCB1B0(**a1, a2, dword_4D0470C, a4);
        }
        if ( (unsigned __int8)(v26 - 6) > 2u && v26 != 13 )
          sub_91B8A0("unsupported float variant!");
      }
      return sub_BCB170(**a1);
    case 6:
      v27 = sub_91A3B0(a1, *(_QWORD *)(v5 + 160));
      return sub_BCE760(v27, 0);
    case 7:
      v21 = *(_BYTE *)(*(_QWORD *)(v5 + 168) + 16LL);
      v22 = sub_9380F0(a1, v5, a3);
      return sub_939CF0(a1, v22, v21 & 1, v5 + 64);
    case 8:
      if ( (*(_BYTE *)(v5 + 169) & 2) != 0 )
        sub_91B8A0("variable length arrays are not supported!");
      v23 = sub_91A3B0(a1, *(_QWORD *)(v5 + 160));
      return sub_BCD420(v23, *(_QWORD *)(v5 + 176));
    case 0xA:
    case 0xB:
      v11 = "struct.";
      v12 = sub_919890((__int64)a1, v5);
      v13 = *(_BYTE *)(v5 + 140) == 10;
      v14 = v12;
      v28 = v30;
      if ( !v13 )
        v11 = "union.";
      v15 = strlen(v11);
      if ( (unsigned int)v15 < 8 )
      {
        if ( (v15 & 4) != 0 )
        {
          LODWORD(v30[0]) = *(_DWORD *)v11;
          v16 = *(unsigned int *)&v11[(unsigned int)v15 - 4];
          *(_DWORD *)((char *)&v29 + (unsigned int)v15 + 4) = v16;
        }
        else if ( (_DWORD)v15 )
        {
          LOBYTE(v30[0]) = *v11;
          if ( (v15 & 2) != 0 )
          {
            v16 = *(unsigned __int16 *)&v11[(unsigned int)v15 - 2];
            *(_WORD *)((char *)&v29 + (unsigned int)v15 + 6) = v16;
          }
        }
      }
      else
      {
        v16 = *(_QWORD *)&v11[(unsigned int)v15 - 8];
        *(_QWORD *)((char *)&v30[-1] + (unsigned int)v15) = v16;
        if ( (unsigned int)(v15 - 1) >= 8 )
        {
          v17 = 0;
          do
          {
            v16 = v17;
            v17 += 8;
            *(_QWORD *)((char *)v30 + v16) = *(_QWORD *)&v11[v16];
          }
          while ( v17 < (((_DWORD)v15 - 1) & 0xFFFFFFF8) );
        }
      }
      v18 = *(const char **)(v5 + 8);
      v29 = v15;
      *((_BYTE *)v30 + v15) = 0;
      if ( !v18 )
      {
        if ( 0x3FFFFFFFFFFFFFFFLL - v29 > 3 )
        {
          sub_2241490(&v28, "anon", 4, v16);
          goto LABEL_21;
        }
LABEL_50:
        sub_4262D8((__int64)"basic_string::append");
      }
      v19 = strlen(v18);
      if ( v19 > 0x3FFFFFFFFFFFFFFFLL - v29 )
        goto LABEL_50;
      sub_2241490(&v28, v18, v19, v20);
LABEL_21:
      sub_BCB4B0(v14, v28, v29);
      if ( v28 != v30 )
        j_j___libc_free_0(v28, v30[0] + 1LL);
      return v14;
    case 0xF:
      return sub_91A3B0(a1, *(_QWORD *)(v5 + 160));
    default:
      goto LABEL_10;
  }
}
