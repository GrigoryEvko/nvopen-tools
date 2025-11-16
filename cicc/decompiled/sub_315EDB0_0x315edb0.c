// Function: sub_315EDB0
// Address: 0x315edb0
//
void __fastcall sub_315EDB0(__int64 a1, __int64 *a2, __int64 **a3)
{
  __int64 *v3; // r13
  unsigned __int64 v6; // rbx
  __int64 *v7; // rdi
  __int64 v8; // rax
  __int64 v9; // rax
  unsigned __int64 v10; // rbx
  __int64 *v11; // r15
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // rbx
  __int64 v16; // r15
  bool v17; // cc
  unsigned __int64 v18; // rdi
  int v19; // [rsp+0h] [rbp-80h]
  int v20; // [rsp+4h] [rbp-7Ch]
  __int64 v21; // [rsp+8h] [rbp-78h]
  int v22; // [rsp+10h] [rbp-70h]
  char v23; // [rsp+17h] [rbp-69h]
  __int64 v24; // [rsp+18h] [rbp-68h]
  _QWORD v25[10]; // [rsp+30h] [rbp-50h] BYREF

  if ( (__int64 *)a1 != a2 )
  {
    v3 = (__int64 *)(a1 + 48);
    while ( a2 != v3 )
    {
      sub_B4CED0((__int64)v25, *v3, **a3);
      v6 = v25[0];
      sub_B4CED0((__int64)v25, *(_QWORD *)a1, **a3);
      v7 = v3;
      v3 += 6;
      if ( v25[0] >= v6 )
      {
        sub_315EB70((__int64)v7, a3);
      }
      else
      {
        v8 = *(v3 - 6);
        ++*(v3 - 5);
        v24 = v8;
        v9 = *(v3 - 4);
        *(v3 - 4) = 0;
        v21 = v9;
        LODWORD(v9) = *((_DWORD *)v3 - 6);
        *((_DWORD *)v3 - 6) = 0;
        v22 = v9;
        LODWORD(v9) = *((_DWORD *)v3 - 5);
        *((_DWORD *)v3 - 5) = 0;
        v20 = v9;
        LODWORD(v9) = *((_DWORD *)v3 - 4);
        *((_DWORD *)v3 - 4) = 0;
        v19 = v9;
        v23 = *((_BYTE *)v3 - 8);
        v10 = 0xAAAAAAAAAAAAAAABLL * (((__int64)v7 - a1) >> 4);
        if ( (__int64)v7 - a1 > 0 )
        {
          v11 = v7 - 6;
          *v7 = *(v7 - 6);
          while ( 1 )
          {
            sub_C7D6A0(v11[8], 32LL * *((unsigned int *)v11 + 20), 8);
            v13 = v11[2];
            ++v11[7];
            ++v11[1];
            v11[8] = v13;
            LODWORD(v13) = *((_DWORD *)v11 + 6);
            v11[2] = 0;
            *((_DWORD *)v11 + 18) = v13;
            LODWORD(v13) = *((_DWORD *)v11 + 7);
            *((_DWORD *)v11 + 6) = 0;
            *((_DWORD *)v11 + 19) = v13;
            LODWORD(v13) = *((_DWORD *)v11 + 8);
            *((_DWORD *)v11 + 7) = 0;
            *((_DWORD *)v11 + 20) = v13;
            LOBYTE(v13) = *((_BYTE *)v11 + 40);
            *((_DWORD *)v11 + 8) = 0;
            *((_BYTE *)v11 + 88) = v13;
            if ( !--v10 )
              break;
            v12 = *(v11 - 6);
            v11 -= 6;
            v11[6] = v12;
          }
        }
        *(_QWORD *)a1 = v24;
        v14 = *(unsigned int *)(a1 + 32);
        if ( (_DWORD)v14 )
        {
          v15 = *(_QWORD *)(a1 + 16);
          v16 = v15 + 32 * v14;
          do
          {
            while ( 1 )
            {
              if ( *(_QWORD *)v15 != -8192 && *(_QWORD *)v15 != -4096 )
              {
                if ( *(_BYTE *)(v15 + 24) )
                {
                  v17 = *(_DWORD *)(v15 + 16) <= 0x40u;
                  *(_BYTE *)(v15 + 24) = 0;
                  if ( !v17 )
                  {
                    v18 = *(_QWORD *)(v15 + 8);
                    if ( v18 )
                      break;
                  }
                }
              }
              v15 += 32;
              if ( v16 == v15 )
                goto LABEL_17;
            }
            j_j___libc_free_0_0(v18);
            v15 += 32;
          }
          while ( v16 != v15 );
LABEL_17:
          LODWORD(v14) = *(_DWORD *)(a1 + 32);
        }
        sub_C7D6A0(*(_QWORD *)(a1 + 16), 32LL * (unsigned int)v14, 8);
        ++*(_QWORD *)(a1 + 8);
        *(_QWORD *)(a1 + 16) = v21;
        *(_DWORD *)(a1 + 24) = v22;
        *(_DWORD *)(a1 + 28) = v20;
        *(_DWORD *)(a1 + 32) = v19;
        *(_BYTE *)(a1 + 40) = v23;
        sub_C7D6A0(0, 0, 8);
      }
    }
  }
}
