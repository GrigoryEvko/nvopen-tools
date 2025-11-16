// Function: sub_1A69990
// Address: 0x1a69990
//
__int64 __fastcall sub_1A69990(__int64 a1, __int64 a2, __int64 a3, __int64 *a4)
{
  char v8; // al
  __int64 v9; // rax
  __int64 v10; // rdi
  __int64 v11; // rsi
  __int64 v12; // r15
  __int64 v13; // rax
  __int64 v14; // r9
  __int64 v15; // r8
  __int64 v16; // rcx
  __int64 v17; // rdx
  __int64 v19; // r15
  __int64 v20; // r10
  __int64 v21; // rax
  __int64 v22; // rax
  __int64 v23; // rax
  int v24; // eax
  __int64 v25; // rcx
  __int64 v26; // rax
  __int64 v27; // [rsp+8h] [rbp-38h]

  v8 = *(_BYTE *)(a2 + 16);
  switch ( v8 )
  {
    case 35:
      v11 = *(_QWORD *)(a2 - 48);
      if ( v11 )
      {
        v12 = *(_QWORD *)(a2 - 24);
        if ( *(_BYTE *)(v12 + 16) == 13 )
          goto LABEL_15;
        if ( *(_BYTE *)(v11 + 16) != 13 )
          break;
      }
      else
      {
        if ( MEMORY[0x10] != 13 )
          break;
        v12 = *(_QWORD *)(a2 - 24);
        if ( !v12 )
          break;
      }
      v22 = v11;
      v11 = v12;
      v12 = v22;
      goto LABEL_15;
    case 5:
      if ( *(_WORD *)(a2 + 18) != 11
        || ((v24 = *(_DWORD *)(a2 + 20), v25 = v24 & 0xFFFFFFF, (v11 = *(_QWORD *)(a2 - 24 * v25)) == 0)
         || (v12 = *(_QWORD *)(a2 + 24 * (1 - v25)), *(_BYTE *)(v12 + 16) != 13))
        && ((v26 = v24 & 0xFFFFFFF, v12 = *(_QWORD *)(a2 - 24 * v26), *(_BYTE *)(v12 + 16) != 13)
         || (v11 = *(_QWORD *)(a2 + 24 * (1 - v26))) == 0) )
      {
        if ( *(_WORD *)(a2 + 18) != 27 )
          break;
        v19 = *(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
        if ( !v19
          || (v20 = *(_QWORD *)(a2 + 24 * (1LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF))), *(_BYTE *)(v20 + 16) != 13) )
        {
          v23 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
          v20 = *(_QWORD *)(a2 - 24 * v23);
          if ( *(_BYTE *)(v20 + 16) != 13 )
            break;
          v19 = *(_QWORD *)(a2 + 24 * (1 - v23));
          if ( !v19 )
            break;
        }
LABEL_11:
        v27 = v20;
        if ( (unsigned __int8)sub_14BB210(v19, v20, *(_QWORD *)(a1 + 160), 0, 0, 0) )
        {
          v21 = sub_146F1B0(*(_QWORD *)(a1 + 176), v19);
          v14 = (__int64)a4;
          v15 = a3;
          v17 = v21;
          v16 = v27;
          return sub_1A69110(a1, 2, v17, v16, v15, v14);
        }
        break;
      }
LABEL_15:
      v10 = *(_QWORD *)(a1 + 176);
      goto LABEL_6;
    case 51:
      v19 = *(_QWORD *)(a2 - 48);
      if ( v19 )
      {
        v20 = *(_QWORD *)(a2 - 24);
        if ( *(_BYTE *)(v20 + 16) == 13 )
          goto LABEL_11;
        if ( *(_BYTE *)(v19 + 16) == 13 )
          goto LABEL_29;
      }
      else if ( MEMORY[0x10] == 13 )
      {
        v20 = *(_QWORD *)(a2 - 24);
        if ( v20 )
        {
LABEL_29:
          v19 = v20;
          v20 = *(_QWORD *)(a2 - 48);
          goto LABEL_11;
        }
      }
      break;
  }
  v9 = sub_159C470(*a4, 0, 0);
  v10 = *(_QWORD *)(a1 + 176);
  v11 = a2;
  v12 = v9;
LABEL_6:
  v13 = sub_146F1B0(v10, v11);
  v14 = (__int64)a4;
  v15 = a3;
  v16 = v12;
  v17 = v13;
  return sub_1A69110(a1, 2, v17, v16, v15, v14);
}
