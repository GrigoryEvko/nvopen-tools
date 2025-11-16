// Function: sub_E33A00
// Address: 0xe33a00
//
void __fastcall sub_E33A00(unsigned __int8 *a1, __int64 a2)
{
  __int64 v3; // rdi
  __int64 v4; // rax
  __int64 v5; // r8
  __int64 v6; // r9
  unsigned __int8 v7; // dl
  __int64 *v8; // r13
  __int64 v9; // rcx
  __int64 v10; // rdx
  unsigned __int8 v11; // al
  __int64 i; // rdx
  __int64 v13; // rax
  unsigned __int64 v14; // rbx
  int v15; // edx
  __int64 v16; // r14
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // r12
  int v20; // r12d
  __int64 v21; // rax
  __int64 v22; // rdx
  __int64 v23; // rcx
  __int64 v24; // rax
  unsigned __int8 *v25; // rbx
  __int64 *v26; // [rsp+0h] [rbp-40h]

  v3 = *((_QWORD *)a1 - 4);
  if ( v3 )
  {
    if ( !*(_BYTE *)v3 && *((_QWORD *)a1 + 10) == *(_QWORD *)(v3 + 24) && (*(_BYTE *)(v3 + 7) & 0x20) != 0 )
    {
      v4 = sub_B91C10(v3, 26);
      if ( v4 )
      {
        v7 = *(_BYTE *)(v4 - 16);
        if ( (v7 & 2) != 0 )
        {
          v8 = *(__int64 **)(v4 - 32);
          v9 = *(unsigned int *)(v4 - 24);
        }
        else
        {
          v9 = (*(_WORD *)(v4 - 16) >> 6) & 0xF;
          v8 = (__int64 *)(v4 - 8LL * ((v7 >> 2) & 0xF) - 16);
        }
        v26 = &v8[v9];
        if ( v8 != v26 )
        {
          v10 = *v8;
          v11 = *(_BYTE *)(*v8 - 16);
          if ( (v11 & 2) == 0 )
            goto LABEL_27;
LABEL_11:
          for ( i = *(_QWORD *)(v10 - 32); ; i = -16 - 8LL * ((v11 >> 2) & 0xF) + v10 )
          {
            v13 = *(_QWORD *)(*(_QWORD *)i + 136LL);
            v14 = *(_QWORD *)(v13 + 24);
            if ( *(_DWORD *)(v13 + 32) > 0x40u )
              v14 = *(_QWORD *)v14;
            v15 = *a1;
            if ( v15 == 40 )
            {
              v16 = 32LL * (unsigned int)sub_B491D0((__int64)a1);
            }
            else
            {
              v16 = 0;
              if ( v15 != 85 )
              {
                if ( v15 != 34 )
                  BUG();
                v16 = 64;
              }
            }
            if ( (a1[7] & 0x80u) == 0 )
              goto LABEL_29;
            v17 = sub_BD2BC0((__int64)a1);
            v19 = v17 + v18;
            if ( (a1[7] & 0x80u) == 0 )
              break;
            if ( !(unsigned int)((v19 - sub_BD2BC0((__int64)a1)) >> 4) )
              goto LABEL_29;
            if ( (a1[7] & 0x80u) == 0 )
              goto LABEL_37;
            v20 = *(_DWORD *)(sub_BD2BC0((__int64)a1) + 8);
            if ( (a1[7] & 0x80u) == 0 )
              BUG();
            v21 = sub_BD2BC0((__int64)a1);
            v23 = 32LL * (unsigned int)(*(_DWORD *)(v21 + v22 - 4) - v20);
LABEL_24:
            if ( (unsigned int)((32LL * (*((_DWORD *)a1 + 1) & 0x7FFFFFF) - 32 - v16 - v23) >> 5) > v14 )
            {
              v24 = *(unsigned int *)(a2 + 8);
              v25 = &a1[32 * (v14 - (*((_DWORD *)a1 + 1) & 0x7FFFFFF))];
              if ( v24 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 12) )
              {
                sub_C8D5F0(a2, (const void *)(a2 + 16), v24 + 1, 8u, v5, v6);
                v24 = *(unsigned int *)(a2 + 8);
              }
              *(_QWORD *)(*(_QWORD *)a2 + 8 * v24) = v25;
              ++*(_DWORD *)(a2 + 8);
            }
            if ( v26 == ++v8 )
              return;
            v10 = *v8;
            v11 = *(_BYTE *)(*v8 - 16);
            if ( (v11 & 2) != 0 )
              goto LABEL_11;
LABEL_27:
            ;
          }
          if ( (unsigned int)(v19 >> 4) )
LABEL_37:
            BUG();
LABEL_29:
          v23 = 0;
          goto LABEL_24;
        }
      }
    }
  }
}
