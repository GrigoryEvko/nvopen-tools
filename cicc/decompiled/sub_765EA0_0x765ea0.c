// Function: sub_765EA0
// Address: 0x765ea0
//
void __fastcall sub_765EA0(__int64 *a1, unsigned int a2, _DWORD *a3)
{
  __int64 *v5; // rbx
  __int64 v6; // rsi
  __int64 v7; // r13
  char v8; // al
  char *v9; // rdi
  __int64 v10; // rax
  __int64 v11; // r14
  char *v12; // rdi
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // [rsp-40h] [rbp-40h]

  if ( a1 )
  {
    v5 = a1;
    do
    {
      v6 = *((unsigned __int8 *)v5 + 16);
      v7 = v5[3];
      if ( (_BYTE)v6 == 55 )
        goto LABEL_23;
      while ( 1 )
      {
        switch ( (_BYTE)v6 )
        {
          case 0x35:
            v9 = *(char **)(v7 + 24);
            v6 = *(unsigned __int8 *)(v7 + 16);
            v10 = 0;
            v11 = 0;
            goto LABEL_16;
          case 0x36:
            v9 = *(char **)(v7 + 16);
            v6 = *(unsigned __int8 *)(v7 + 8);
            v11 = v7;
            v10 = 0;
            v7 = 0;
            goto LABEL_16;
          case 0x38:
            v9 = *(char **)(v7 + 16);
            v6 = *(unsigned __int8 *)(v7 + 8);
            v10 = v7;
            v11 = 0;
            v7 = 0;
LABEL_16:
            if ( a2 )
            {
              if ( v9 )
                goto LABEL_18;
            }
            else if ( *(v9 - 8) >= 0 )
            {
              goto LABEL_21;
            }
LABEL_19:
            *((_BYTE *)v5 - 8) |= 0x80u;
            if ( v11 )
            {
              *(_BYTE *)(v11 - 8) |= 0x80u;
            }
            else if ( v10 )
            {
              *(_BYTE *)(v10 - 8) |= 0x80u;
            }
            else if ( v7 )
            {
              v12 = *(char **)(v7 + 32);
              *(_BYTE *)(v7 - 8) |= 0x80u;
              if ( v12 )
              {
                if ( *(v12 - 8) >= 0 )
                {
                  sub_760BD0(v12, 6);
                  if ( !a2 )
                    *a3 = 1;
                }
              }
            }
            goto LABEL_21;
          case 6:
            if ( *(_BYTE *)(v7 + 140) != 12
              || (v13 = *(_QWORD *)(v7 + 8)) == 0
              || (v14 = *(_QWORD *)(v7 + 160), (unsigned __int8)(*(_BYTE *)(v14 + 140) - 9) > 2u)
              || (*(_BYTE *)(v14 + 177) & 4) == 0
              || *(_QWORD *)(v14 + 8) != v13
              || *(char *)(v14 - 8) >= 0 )
            {
              if ( a2 )
              {
LABEL_32:
                sub_760BD0((_QWORD *)v7, 6);
                *((_BYTE *)v5 - 8) |= 0x80u;
                goto LABEL_21;
              }
LABEL_49:
              if ( *(char *)(v7 - 8) >= 0 )
                goto LABEL_21;
LABEL_35:
              *((_BYTE *)v5 - 8) |= 0x80u;
              goto LABEL_21;
            }
            *(_BYTE *)(v7 - 8) |= 0x80u;
            if ( a2 )
              goto LABEL_32;
            v8 = *((_BYTE *)v5 + 16);
            break;
          default:
            v8 = v6;
            if ( a2 )
            {
              if ( !v7 )
                goto LABEL_35;
              v9 = (char *)v7;
              v10 = 0;
              v11 = 0;
              v7 = 0;
LABEL_18:
              v15 = v10;
              sub_760BD0(v9, v6);
              v10 = v15;
              goto LABEL_19;
            }
            break;
        }
        if ( v8 != 69 )
          goto LABEL_49;
        if ( *(char *)(v5[3] - 8) >= 0 )
          break;
        while ( 1 )
        {
LABEL_21:
          v5 = (__int64 *)*v5;
          if ( !v5 )
            return;
          v6 = *((unsigned __int8 *)v5 + 16);
          v7 = v5[3];
          if ( (_BYTE)v6 != 55 )
            break;
LABEL_23:
          sub_765EA0(*(_QWORD *)(v7 + 8), a2, a3);
        }
      }
      if ( v7 )
        sub_760BD0((_QWORD *)v7, v6);
      *a3 = 1;
      *((_BYTE *)v5 - 8) |= 0x80u;
      v5 = (__int64 *)*v5;
    }
    while ( v5 );
  }
}
