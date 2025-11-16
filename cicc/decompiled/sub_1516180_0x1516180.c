// Function: sub_1516180
// Address: 0x1516180
//
void __fastcall sub_1516180(__int64 *a1, __int64 a2)
{
  __int64 v2; // r15
  __int64 v3; // r14
  __int64 v4; // rbx
  __int64 j; // r12
  __int64 v6; // rax
  __int64 v7; // rdx
  _QWORD *v8; // rcx
  __int64 v9; // rax
  __int64 v10; // r9
  _BYTE *v11; // rax
  _BYTE *v12; // r9
  size_t v13; // r11
  __int64 v14; // r8
  _BYTE *v15; // rdi
  int v16; // edx
  _BYTE *v17; // rsi
  __int64 v18; // rdi
  __int64 v19; // rdx
  __int64 v20; // rax
  __int64 v21; // rax
  __int64 *v22; // rdx
  __int64 v23; // rsi
  unsigned __int64 v24; // rcx
  __int64 v25; // rcx
  __int64 v26; // rdi
  _BYTE *v27; // [rsp-B8h] [rbp-B8h]
  size_t v28; // [rsp-B0h] [rbp-B0h]
  _BYTE *v29; // [rsp-A8h] [rbp-A8h]
  __int64 v30; // [rsp-A0h] [rbp-A0h]
  __int64 v31; // [rsp-98h] [rbp-98h]
  int v32; // [rsp-98h] [rbp-98h]
  __int64 i; // [rsp-90h] [rbp-90h]
  _BYTE *v34; // [rsp-88h] [rbp-88h] BYREF
  __int64 v35; // [rsp-80h] [rbp-80h]
  _BYTE v36[120]; // [rsp-78h] [rbp-78h] BYREF

  v26 = *a1;
  if ( *(_BYTE *)(v26 + 1011) )
  {
    v2 = v26;
    v3 = *(_QWORD *)(a2 + 80);
    for ( i = a2 + 72; i != v3; v3 = *(_QWORD *)(v3 + 8) )
    {
      if ( !v3 )
        BUG();
      v4 = *(_QWORD *)(v3 + 24);
      for ( j = v3 + 16; j != v4; v4 = *(_QWORD *)(v4 + 8) )
      {
        while ( 1 )
        {
          if ( !v4 )
            BUG();
          if ( *(_BYTE *)(v4 - 8) == 78 )
          {
            v6 = *(_QWORD *)(v4 - 48);
            if ( !*(_BYTE *)(v6 + 16) && (*(_BYTE *)(v6 + 33) & 0x20) != 0 && *(_DWORD *)(v6 + 36) == 36 )
            {
              v7 = *(_QWORD *)(*(_QWORD *)(v4 - 24 + 24 * (2LL - (*(_DWORD *)(v4 - 4) & 0xFFFFFFF))) + 24LL);
              if ( v7 )
              {
                v8 = *(_QWORD **)(v7 + 24);
                if ( (unsigned int)((__int64)(*(_QWORD *)(v7 + 32) - (_QWORD)v8) >> 3) )
                {
                  if ( *v8 == 6 )
                  {
                    v31 = *(_QWORD *)(*(_QWORD *)(v4 - 24 + 24 * (2LL - (*(_DWORD *)(v4 - 4) & 0xFFFFFFF))) + 24LL);
                    v9 = sub_1601A30(v4 - 24, 1);
                    if ( v9 )
                    {
                      if ( *(_BYTE *)(v9 + 16) == 17 )
                      {
                        v35 = 0x800000000LL;
                        v10 = *(_QWORD *)(v31 + 24);
                        v11 = *(_BYTE **)(v31 + 32);
                        v34 = v36;
                        v12 = (_BYTE *)(v10 + 8);
                        v13 = v11 - v12;
                        v14 = (v11 - v12) >> 3;
                        if ( (unsigned __int64)(v11 - v12) > 0x40 )
                        {
                          v27 = v12;
                          v28 = v11 - v12;
                          v29 = v11;
                          v30 = (v11 - v12) >> 3;
                          sub_16CD150(&v34, v36, v30, 8);
                          v17 = v34;
                          v16 = v35;
                          LODWORD(v14) = v30;
                          v11 = v29;
                          v13 = v28;
                          v15 = &v34[8 * (unsigned int)v35];
                          v12 = v27;
                        }
                        else
                        {
                          v15 = v36;
                          v16 = 0;
                          v17 = v36;
                        }
                        if ( v11 != v12 )
                        {
                          v32 = v14;
                          memcpy(v15, v12, v13);
                          v17 = v34;
                          v16 = v35;
                          LODWORD(v14) = v32;
                        }
                        v18 = *(_QWORD *)(v2 + 240);
                        v19 = (unsigned int)(v16 + v14);
                        LODWORD(v35) = v19;
                        v20 = sub_15C4420(v18, v17, v19, 0, 1);
                        v21 = sub_1628DA0(*(_QWORD *)(v2 + 240), v20);
                        v22 = (__int64 *)(v4 - 24 + 24 * (2LL - (*(_DWORD *)(v4 - 4) & 0xFFFFFFF)));
                        if ( *v22 )
                        {
                          v23 = v22[1];
                          v24 = v22[2] & 0xFFFFFFFFFFFFFFFCLL;
                          *(_QWORD *)v24 = v23;
                          if ( v23 )
                            *(_QWORD *)(v23 + 16) = *(_QWORD *)(v23 + 16) & 3LL | v24;
                        }
                        *v22 = v21;
                        if ( v21 )
                        {
                          v25 = *(_QWORD *)(v21 + 8);
                          v22[1] = v25;
                          if ( v25 )
                            *(_QWORD *)(v25 + 16) = (unsigned __int64)(v22 + 1) | *(_QWORD *)(v25 + 16) & 3LL;
                          v22[2] = (v21 + 8) | v22[2] & 3;
                          *(_QWORD *)(v21 + 8) = v22;
                        }
                        if ( v34 != v36 )
                          break;
                      }
                    }
                  }
                }
              }
            }
          }
          v4 = *(_QWORD *)(v4 + 8);
          if ( j == v4 )
            goto LABEL_30;
        }
        _libc_free((unsigned __int64)v34);
      }
LABEL_30:
      ;
    }
  }
  else
  {
    nullsub_2018();
  }
}
