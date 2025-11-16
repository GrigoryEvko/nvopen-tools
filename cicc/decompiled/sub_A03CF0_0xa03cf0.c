// Function: sub_A03CF0
// Address: 0xa03cf0
//
void __fastcall sub_A03CF0(__int64 *a1, __int64 a2)
{
  _QWORD *v2; // rcx
  _QWORD *j; // r12
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // r15
  __int64 v7; // rbx
  __int64 v8; // rax
  __int64 v9; // r14
  _BYTE *v10; // rax
  __int64 v11; // r9
  _BYTE *v12; // r9
  _BYTE *v13; // rax
  size_t v14; // r10
  __int64 v15; // r8
  _BYTE *v16; // rsi
  int v17; // edx
  _BYTE *v18; // rdi
  __int64 v19; // rax
  __int64 v20; // rcx
  __int64 v21; // r9
  __int64 v22; // rsi
  __int64 v23; // rax
  __int64 v24; // rdi
  _BYTE *v25; // [rsp-E0h] [rbp-E0h]
  __int64 v26; // [rsp-D8h] [rbp-D8h]
  _BYTE *v27; // [rsp-D0h] [rbp-D0h]
  _QWORD *v28; // [rsp-C8h] [rbp-C8h]
  _QWORD *i; // [rsp-C0h] [rbp-C0h]
  int v30; // [rsp-B8h] [rbp-B8h]
  __int64 v31; // [rsp-98h] [rbp-98h] BYREF
  __int64 v32; // [rsp-90h] [rbp-90h] BYREF
  _BYTE *v33; // [rsp-88h] [rbp-88h] BYREF
  __int64 v34; // [rsp-80h] [rbp-80h]
  _BYTE v35[120]; // [rsp-78h] [rbp-78h] BYREF

  v24 = *a1;
  if ( *(_BYTE *)(v24 + 1099) )
  {
    v2 = *(_QWORD **)(a2 + 80);
    v31 = v24;
    v28 = (_QWORD *)(a2 + 72);
    for ( i = v2; v28 != i; i = (_QWORD *)i[1] )
    {
      if ( !i )
        BUG();
      for ( j = (_QWORD *)i[4]; i + 3 != j; j = (_QWORD *)j[1] )
      {
        if ( !j )
          BUG();
        if ( j[5] )
        {
          v4 = sub_B14240();
          v6 = v5;
          v7 = v4;
          if ( v4 != v5 )
          {
            while ( *(_BYTE *)(v7 + 32) )
            {
              v7 = *(_QWORD *)(v7 + 8);
              if ( v7 == v5 )
                goto LABEL_16;
            }
LABEL_11:
            if ( v7 != v6 )
            {
              if ( !*(_BYTE *)(v7 + 64) )
              {
                v8 = sub_B11F60(v7 + 80);
                v9 = v8;
                if ( v8 )
                {
                  if ( (unsigned __int8)sub_AF4730(v8) )
                  {
                    v10 = (_BYTE *)sub_B13320(v7);
                    if ( v10 )
                    {
                      if ( *v10 == 22 )
                      {
                        v11 = *(_QWORD *)(v9 + 16);
                        v33 = v35;
                        v12 = (_BYTE *)(v11 + 8);
                        v34 = 0x800000000LL;
                        v13 = *(_BYTE **)(v9 + 24);
                        v14 = v13 - v12;
                        v15 = (v13 - v12) >> 3;
                        if ( (unsigned __int64)(v13 - v12) > 0x40 )
                        {
                          v25 = v12;
                          v26 = *(_QWORD *)(v9 + 24) - (_QWORD)v12;
                          v27 = *(_BYTE **)(v9 + 24);
                          sub_C8D5F0(&v33, v35, v26 >> 3, 8);
                          v16 = v33;
                          v17 = v34;
                          LODWORD(v15) = v26 >> 3;
                          v13 = v27;
                          v14 = v26;
                          v12 = v25;
                          v18 = &v33[8 * (unsigned int)v34];
                        }
                        else
                        {
                          v16 = v35;
                          v17 = 0;
                          v18 = v35;
                        }
                        if ( v13 != v12 )
                        {
                          v30 = v15;
                          memcpy(v18, v12, v14);
                          v16 = v33;
                          v17 = v34;
                          LODWORD(v15) = v30;
                        }
                        LODWORD(v34) = v15 + v17;
                        v19 = sub_B0D000(*(_QWORD *)(v31 + 248), v16, (unsigned int)(v15 + v17), 0, 1);
                        sub_B11F20(&v32, v19);
                        if ( *(_QWORD *)(v7 + 80) )
                          sub_B91220(v7 + 80);
                        v22 = v32;
                        *(_QWORD *)(v7 + 80) = v32;
                        if ( v22 )
                          sub_B976B0(&v32, v22, v7 + 80, v20, &v32, v21);
                        if ( v33 != v35 )
                          _libc_free(v33, v22);
                      }
                    }
                  }
                }
              }
              while ( 1 )
              {
                v7 = *(_QWORD *)(v7 + 8);
                if ( v7 == v6 )
                  break;
                if ( !*(_BYTE *)(v7 + 32) )
                  goto LABEL_11;
              }
            }
          }
        }
LABEL_16:
        if ( *((_BYTE *)j - 24) == 85 )
        {
          v23 = *(j - 7);
          if ( v23 )
          {
            if ( !*(_BYTE *)v23
              && *(_QWORD *)(v23 + 24) == j[7]
              && (*(_BYTE *)(v23 + 33) & 0x20) != 0
              && *(_DWORD *)(v23 + 36) == 69 )
            {
              sub_A02A70(&v31, (__int64)(j - 3));
            }
          }
        }
      }
    }
  }
  else
  {
    nullsub_2008();
  }
}
