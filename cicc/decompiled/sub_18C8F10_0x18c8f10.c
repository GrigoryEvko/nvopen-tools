// Function: sub_18C8F10
// Address: 0x18c8f10
//
__int64 __fastcall sub_18C8F10(__int64 a1, __int64 *a2)
{
  unsigned int v2; // r15d
  __int64 v4; // rax
  __int64 v5; // r13
  __int64 v6; // rax
  __int64 v7; // r14
  __int64 v8; // rsi
  __int64 v9; // rbx
  int v10; // eax
  int v11; // ecx
  __int64 v12; // r9
  __int64 v13; // rbx
  __int64 v14; // r9
  _QWORD *v15; // r10
  __int64 v16; // r12
  unsigned __int8 v17; // al
  __int64 v18; // r11
  __int64 v19; // rdi
  int v20; // eax
  _QWORD *v21; // rsi
  unsigned __int8 v22; // al
  __int64 v23; // rdi
  char v24; // al
  _QWORD *v25; // [rsp+0h] [rbp-50h]
  _QWORD *v26; // [rsp+8h] [rbp-48h]
  __int64 v27; // [rsp+10h] [rbp-40h]
  __int64 v28; // [rsp+10h] [rbp-40h]
  unsigned __int8 v29; // [rsp+18h] [rbp-38h]
  unsigned __int8 v30; // [rsp+18h] [rbp-38h]

  if ( unk_4F99CA8 )
  {
    if ( sub_18C8CE0((__int64)a2) )
    {
      v2 = sub_1636800(a1, a2);
      if ( !(_BYTE)v2 )
      {
        v4 = sub_16321C0((__int64)a2, (__int64)"llvm.global_ctors", 17, 0);
        if ( v4 )
        {
          v5 = *(_QWORD *)(v4 - 24);
          v6 = 24LL * (*(_DWORD *)(v5 + 20) & 0xFFFFFFF);
          v7 = v5 - v6;
          if ( v5 != v5 - v6 )
          {
            while ( 1 )
            {
              v8 = 1LL - (*(_DWORD *)(*(_QWORD *)v7 + 20LL) & 0xFFFFFFF);
              v9 = *(_QWORD *)(*(_QWORD *)v7 + 24 * v8);
              if ( !*(_BYTE *)(v9 + 16) )
              {
                LOBYTE(v10) = sub_15E4F60(*(_QWORD *)(*(_QWORD *)v7 + 24 * v8));
                v11 = v10;
                if ( !(_BYTE)v10 )
                {
                  v12 = *(_QWORD *)(v9 + 80);
                  if ( *(_QWORD *)(v12 + 8) == v9 + 72 )
                  {
                    v13 = *(_QWORD *)(v12 + 24);
                    v14 = v12 + 16;
                    v15 = 0;
                    if ( v13 != v14 )
                      break;
                  }
                }
              }
LABEL_9:
              v7 += 24;
              if ( v5 == v7 )
                return v2;
            }
            while ( 1 )
            {
              v16 = v13;
              v13 = *(_QWORD *)(v13 + 8);
              v17 = *(_BYTE *)(v16 - 8);
              if ( v17 <= 0x17u )
                goto LABEL_16;
              v18 = v16 - 24;
              if ( v17 == 78 )
              {
                v19 = *(_QWORD *)(v16 - 48);
                if ( *(_BYTE *)(v19 + 16) )
                  goto LABEL_37;
                v25 = v15;
                v27 = v14;
                v29 = v11;
                v20 = sub_1438F00(v19);
                v11 = v29;
                v14 = v27;
                v18 = v16 - 24;
                v15 = v25;
                switch ( v20 )
                {
                  case 8:
                    if ( v25 )
                    {
                      v21 = *(_QWORD **)(v16 - 24LL * (*(_DWORD *)(v16 - 4) & 0xFFFFFFF) - 24);
                      if ( v21 != 0 && v21 == v25 )
                      {
                        sub_15F20C0((_QWORD *)(v16 - 24));
                        sub_15F20C0(v25);
                        v14 = v27;
                        LOBYTE(v16) = v21 != 0 && v21 == v25;
                        v11 = v16;
                      }
                      v15 = 0;
                    }
                    goto LABEL_16;
                  case 21:
                    v22 = *(_BYTE *)(v16 - 8);
                    if ( v22 <= 0x17u )
                      goto LABEL_33;
                    if ( v22 == 78 )
                    {
LABEL_37:
                      v23 = v18 | 4;
                    }
                    else if ( v22 == 29 )
                    {
LABEL_38:
                      v23 = v18 & 0xFFFFFFFFFFFFFFFBLL;
                    }
                    else
                    {
LABEL_33:
                      v23 = 0;
                    }
                    v26 = v15;
                    v28 = v14;
                    v30 = v11;
                    v24 = sub_18C8950(v23, 0);
                    v15 = v26;
                    v11 = v30;
                    v14 = v28;
                    if ( v24 )
                      v15 = 0;
                    goto LABEL_16;
                  case 7:
                    v15 = (_QWORD *)(v16 - 24);
                    break;
                }
                if ( v27 == v13 )
                {
LABEL_25:
                  v2 |= v11;
                  goto LABEL_9;
                }
              }
              else
              {
                if ( v17 == 29 )
                  goto LABEL_38;
LABEL_16:
                if ( v14 == v13 )
                  goto LABEL_25;
              }
            }
          }
        }
      }
    }
  }
  return 0;
}
