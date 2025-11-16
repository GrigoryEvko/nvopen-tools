// Function: sub_1279AB0
// Address: 0x1279ab0
//
__int64 __fastcall sub_1279AB0(_QWORD **a1, __int64 a2, unsigned __int8 a3)
{
  __int64 v4; // rax
  __int64 v5; // rax
  __int64 v6; // r14
  __int64 v7; // rax
  const char *v9; // r14
  __int64 v10; // rax
  bool v11; // zf
  __int64 v12; // r12
  size_t v13; // rax
  __int64 v14; // rcx
  unsigned int v15; // edx
  const char *v16; // r14
  size_t v17; // rdx
  __int64 v18; // rcx
  char v19; // r14
  __int64 v20; // rax
  __int64 v21; // rax
  _QWORD *v22; // rax
  __int64 v23; // rsi
  char v24; // al
  __int64 v25; // rax
  _QWORD *v26; // [rsp+0h] [rbp-40h] BYREF
  size_t v27; // [rsp+8h] [rbp-38h]
  _QWORD v28[6]; // [rsp+10h] [rbp-30h] BYREF

  if ( *(_BYTE *)(a2 + 140) == 12 )
  {
    v4 = a2;
    do
      v4 = *(_QWORD *)(v4 + 160);
    while ( *(_BYTE *)(v4 + 140) == 12 );
    if ( a2 != v4 )
      sub_127B550("internal error while translating type!");
  }
  if ( (*(_BYTE *)(a2 + 89) & 1) != 0 )
  {
    v5 = sub_72B7D0(a2);
    v6 = v5;
    if ( v5 )
    {
      v7 = sub_72B840(v5);
      sub_5E3920(a2, v6, *(_DWORD *)(v7 + 24));
      if ( *(_BYTE *)(a2 + 140) > 0xFu )
LABEL_10:
        sub_127B550("unsupported type during translation!");
    }
  }
  switch ( *(_BYTE *)(a2 + 140) )
  {
    case 1:
      v22 = *a1;
      v23 = 8;
      return sub_1644900(*v22, v23);
    case 2:
      v22 = *a1;
      v23 = 8 * (unsigned int)*(_QWORD *)(a2 + 128);
      return sub_1644900(*v22, v23);
    case 3:
      v24 = *(_BYTE *)(a2 + 160);
      if ( v24 == 2 )
        return sub_16432A0(**a1);
      if ( v24 != 4 && (unsigned __int8)(v24 - 6) > 2u && v24 != 13 )
        sub_127B550("unsupported float variant!");
      return sub_16432B0(**a1);
    case 6:
      v25 = sub_127A050(a1, *(_QWORD *)(a2 + 160));
      return sub_1646BA0(v25, 0);
    case 7:
      v19 = *(_BYTE *)(*(_QWORD *)(a2 + 168) + 16LL);
      v20 = sub_1297B70(a1, a2, a3);
      return sub_1299060(a1, v20, v19 & 1, a2 + 64);
    case 8:
      if ( (*(_BYTE *)(a2 + 169) & 2) != 0 )
        sub_127B550("variable length arrays are not supported!");
      v21 = sub_127A050(a1, *(_QWORD *)(a2 + 160));
      return sub_1645D80(v21, *(_QWORD *)(a2 + 176));
    case 0xA:
    case 0xB:
      v9 = "struct.";
      v10 = sub_12794A0((__int64)a1, a2);
      v11 = *(_BYTE *)(a2 + 140) == 10;
      v12 = v10;
      v26 = v28;
      if ( !v11 )
        v9 = "union.";
      v13 = strlen(v9);
      if ( (unsigned int)v13 < 8 )
      {
        if ( (v13 & 4) != 0 )
        {
          LODWORD(v28[0]) = *(_DWORD *)v9;
          v14 = *(unsigned int *)&v9[(unsigned int)v13 - 4];
          *(_DWORD *)((char *)&v27 + (unsigned int)v13 + 4) = v14;
        }
        else if ( (_DWORD)v13 )
        {
          LOBYTE(v28[0]) = *v9;
          if ( (v13 & 2) != 0 )
          {
            v14 = *(unsigned __int16 *)&v9[(unsigned int)v13 - 2];
            *(_WORD *)((char *)&v27 + (unsigned int)v13 + 6) = v14;
          }
        }
      }
      else
      {
        v14 = *(_QWORD *)&v9[(unsigned int)v13 - 8];
        *(_QWORD *)((char *)&v28[-1] + (unsigned int)v13) = v14;
        if ( (unsigned int)(v13 - 1) >= 8 )
        {
          v15 = 0;
          do
          {
            v14 = v15;
            v15 += 8;
            *(_QWORD *)((char *)v28 + v14) = *(_QWORD *)&v9[v14];
          }
          while ( v15 < (((_DWORD)v13 - 1) & 0xFFFFFFF8) );
        }
      }
      v16 = *(const char **)(a2 + 8);
      v27 = v13;
      *((_BYTE *)v28 + v13) = 0;
      if ( !v16 )
      {
        if ( 0x3FFFFFFFFFFFFFFFLL - v27 > 3 )
        {
          sub_2241490(&v26, "anon", 4, v14);
          goto LABEL_21;
        }
LABEL_46:
        sub_4262D8((__int64)"basic_string::append");
      }
      v17 = strlen(v16);
      if ( v17 > 0x3FFFFFFFFFFFFFFFLL - v27 )
        goto LABEL_46;
      sub_2241490(&v26, v16, v17, v18);
LABEL_21:
      sub_1643660(v12, v26, v27);
      if ( v26 != v28 )
        j_j___libc_free_0(v26, v28[0] + 1LL);
      return v12;
    case 0xF:
      return sub_127A050(a1, *(_QWORD *)(a2 + 160));
    default:
      goto LABEL_10;
  }
}
