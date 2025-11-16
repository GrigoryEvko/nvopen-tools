// Function: sub_898C50
// Address: 0x898c50
//
void __fastcall sub_898C50(__int64 a1, char *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rdx
  __int64 *v7; // r13
  __int64 v8; // r12
  __int64 v9; // rax
  unsigned __int8 v10; // al
  __int64 j; // rdx
  __int64 i; // rdx

  if ( a2 )
  {
    v6 = (unsigned int)(unk_4F076E0 | dword_4D04278);
    if ( unk_4F076E0 | dword_4D04278 )
    {
      v7 = (__int64 *)*((_QWORD *)a2 + 1);
      if ( v7 )
      {
        v8 = 0;
        while ( 1 )
        {
          v10 = *((_BYTE *)v7 + 26);
          if ( v10 == 3 )
            goto LABEL_8;
          if ( v10 == 6 )
          {
            a2 = (char *)v7[6];
            v9 = 0;
            for ( i = *a2; (_BYTE)i; i = *a2 )
            {
              ++a2;
              a4 = 32 * v9;
              v9 += 32 * v9 + i;
            }
            goto LABEL_7;
          }
          if ( v10 > 6u )
          {
            if ( v10 != 8 )
              goto LABEL_19;
            v9 = (unsigned int)sub_72DB90(v7[7], (__int64)a2, v6, a4, a5, a6);
          }
          else
          {
            if ( v10 == 1 )
            {
              a2 = *(char **)(v7[6] + 8);
              v9 = 0;
              for ( j = *a2; (_BYTE)j; j = *a2 )
              {
                ++a2;
                a4 = 32 * v9;
                v9 += 32 * v9 + j;
              }
              goto LABEL_7;
            }
            if ( v10 != 2 )
            {
LABEL_19:
              v9 = *((unsigned __int16 *)v7 + 12);
              goto LABEL_7;
            }
            v9 = (unsigned int)sub_72DB90(v7[6], (__int64)a2, v6, a4, a5, a6);
          }
LABEL_7:
          v6 = 9 * v8;
          v8 = v9 + 73 * v8;
LABEL_8:
          v7 = (__int64 *)*v7;
          if ( !v7 )
            goto LABEL_24;
        }
      }
      LODWORD(v8) = 0;
LABEL_24:
      *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 104) + 208LL) + 124LL) = v8;
    }
  }
}
