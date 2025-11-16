// Function: sub_185B5B0
// Address: 0x185b5b0
//
__int64 __fastcall sub_185B5B0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // rax
  __int64 v5; // rbx
  __int64 v6; // rax
  __int64 v7; // rdi
  unsigned int v8; // eax
  unsigned int v9; // r12d
  __int64 v11; // r13
  __int64 v12; // r14
  __int64 v13; // rax
  unsigned __int64 v14; // rax
  char v15; // dl
  unsigned __int64 v16; // r15
  __int64 v17; // rdi
  __int64 *v18; // r13
  __int64 v19; // rsi
  _QWORD *v20; // rax
  unsigned __int64 v21; // rdi
  __int64 v22; // rax
  char v23; // cl
  _QWORD *v24; // rdi
  unsigned __int8 v25; // al
  char v26; // al
  unsigned __int64 v27; // rdi

  v4 = *(_DWORD *)(a1 + 20) & 0xFFFFFFF;
  if ( (unsigned int)v4 > 2 )
  {
    v5 = a1;
    if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
    {
      v6 = *(_QWORD *)(a1 - 8);
    }
    else
    {
      a3 = 24 * v4;
      v6 = a1 - 24 * v4;
    }
    v7 = *(_QWORD *)(v6 + 24);
    if ( *(_BYTE *)(v7 + 16) <= 0x10u )
    {
      LOBYTE(v8) = sub_1593BB0(v7, a2, a3, a4);
      v9 = v8;
      if ( (_BYTE)v8 )
      {
        if ( (*(_BYTE *)(v5 + 23) & 0x40) != 0 )
          v11 = *(_QWORD *)(v5 - 8);
        else
          v11 = v5 - 24LL * (*(_DWORD *)(v5 + 20) & 0xFFFFFFF);
        v12 = v5;
        v13 = sub_16348C0(v5);
        if ( (*(_BYTE *)(v5 + 23) & 0x40) != 0 )
          v12 = *(_QWORD *)(v5 - 8) + 24LL * (*(_DWORD *)(v5 + 20) & 0xFFFFFFF);
        v14 = v13 & 0xFFFFFFFFFFFFFFF8LL;
        if ( !v14 )
          v14 = sub_1643D30(0, *(_QWORD *)(v11 + 24));
        v15 = *(_BYTE *)(v14 + 8);
        if ( ((v15 - 14) & 0xFD) != 0 )
        {
          v17 = 0;
          v16 = -1;
          if ( v15 == 13 )
            v17 = v14;
        }
        else
        {
          v16 = *(_QWORD *)(v14 + 32);
          v17 = *(_QWORD *)(v14 + 24) | 4LL;
        }
        v18 = (__int64 *)(v11 + 48);
        if ( (__int64 *)v12 != v18 )
        {
          while ( 1 )
          {
            v19 = *v18;
            if ( (v17 & 4) != 0 )
            {
              if ( *(_BYTE *)(v19 + 16) != 13 )
                return 0;
              if ( v16 != -1 )
              {
                v20 = *(_QWORD **)(v19 + 24);
                if ( *(_DWORD *)(v19 + 32) > 0x40u )
                  v20 = (_QWORD *)*v20;
                if ( (unsigned __int64)v20 >= v16 )
                  return 0;
              }
              v21 = v17 & 0xFFFFFFFFFFFFFFF8LL;
              v22 = v21;
              if ( v21 )
              {
                v23 = *(_BYTE *)(v21 + 8);
                if ( ((v23 - 14) & 0xFD) != 0 )
                  goto LABEL_35;
                goto LABEL_26;
              }
              v27 = 0;
            }
            else
            {
              v27 = v17 & 0xFFFFFFFFFFFFFFF8LL;
            }
            v22 = sub_1643D30(v27, v19);
            v23 = *(_BYTE *)(v22 + 8);
            if ( ((v23 - 14) & 0xFD) != 0 )
            {
LABEL_35:
              v17 = 0;
              if ( v23 == 13 )
                v17 = v22;
              goto LABEL_27;
            }
LABEL_26:
            v16 = *(_QWORD *)(v22 + 32);
            v17 = *(_QWORD *)(v22 + 24) | 4LL;
LABEL_27:
            v18 += 3;
            if ( (__int64 *)v12 == v18 )
              goto LABEL_28;
          }
        }
        do
        {
          do
          {
            while ( 1 )
            {
LABEL_28:
              v5 = *(_QWORD *)(v5 + 8);
              if ( !v5 )
                return v9;
              v24 = sub_1648700(v5);
              v25 = *((_BYTE *)v24 + 16);
              if ( v25 > 0x10u )
                break;
              v26 = sub_1ACF050(v24);
LABEL_31:
              if ( !v26 )
                return 0;
            }
            if ( v25 <= 0x17u )
              return 0;
          }
          while ( v25 == 54 );
          if ( v25 == 55 )
          {
            v26 = *(v24 - 6) != (_QWORD)v24;
            goto LABEL_31;
          }
        }
        while ( v25 == 56 && (unsigned __int8)sub_185B5B0(v24) );
      }
    }
  }
  return 0;
}
