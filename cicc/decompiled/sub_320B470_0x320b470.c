// Function: sub_320B470
// Address: 0x320b470
//
__int64 __fastcall sub_320B470(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rsi
  __int64 v4; // r13
  __int64 v5; // r15
  __int64 v6; // r12
  __int64 v7; // rdi
  __int64 v8; // rcx
  __int64 v9; // rdx
  unsigned __int64 v10; // r14
  unsigned __int64 v11; // rbx
  unsigned __int64 v12; // rdi
  bool v13; // zf
  unsigned int v14; // edx
  unsigned __int64 v15; // rdi
  unsigned __int64 v17; // rdi
  int v18; // edx
  unsigned __int64 v19; // rdi
  __int64 v20; // rdx

  v3 = a2 - a1;
  v4 = 0x2E8BA2E8BA2E8BA3LL * (v3 >> 3);
  if ( v3 > 0 )
  {
    v5 = a1 + 40;
    v6 = a3;
    while ( 1 )
    {
      while ( 1 )
      {
        v7 = *(_QWORD *)(v6 + 16);
        *(_QWORD *)v6 = *(_QWORD *)(v5 - 40);
        sub_C7D6A0(v7, 12LL * *(unsigned int *)(v6 + 32), 4);
        ++*(_QWORD *)(v6 + 8);
        *(_DWORD *)(v6 + 32) = 0;
        *(_QWORD *)(v6 + 16) = 0;
        *(_DWORD *)(v6 + 24) = 0;
        *(_DWORD *)(v6 + 28) = 0;
        v8 = *(_QWORD *)(v5 - 24);
        ++*(_QWORD *)(v5 - 32);
        v9 = *(_QWORD *)(v6 + 16);
        *(_QWORD *)(v6 + 16) = v8;
        LODWORD(v8) = *(_DWORD *)(v5 - 16);
        *(_QWORD *)(v5 - 24) = v9;
        LODWORD(v9) = *(_DWORD *)(v6 + 24);
        *(_DWORD *)(v6 + 24) = v8;
        LODWORD(v8) = *(_DWORD *)(v5 - 12);
        *(_DWORD *)(v5 - 16) = v9;
        LODWORD(v9) = *(_DWORD *)(v6 + 28);
        *(_DWORD *)(v6 + 28) = v8;
        LODWORD(v8) = *(_DWORD *)(v5 - 8);
        *(_DWORD *)(v5 - 12) = v9;
        LODWORD(v9) = *(_DWORD *)(v6 + 32);
        *(_DWORD *)(v6 + 32) = v8;
        *(_DWORD *)(v5 - 8) = v9;
        if ( v5 != v6 + 40 )
        {
          v10 = *(_QWORD *)(v6 + 40);
          v11 = v10 + 40LL * *(unsigned int *)(v6 + 48);
          if ( *(_DWORD *)(v5 + 8) )
          {
            if ( v11 != v10 )
            {
              do
              {
                v11 -= 40LL;
                v12 = *(_QWORD *)(v11 + 8);
                if ( v12 != v11 + 24 )
                  _libc_free(v12);
              }
              while ( v11 != v10 );
              v10 = *(_QWORD *)(v6 + 40);
            }
            if ( v10 != v6 + 56 )
              _libc_free(v10);
            *(_QWORD *)(v6 + 40) = *(_QWORD *)v5;
            *(_DWORD *)(v6 + 48) = *(_DWORD *)(v5 + 8);
            *(_DWORD *)(v6 + 52) = *(_DWORD *)(v5 + 12);
            *(_QWORD *)v5 = v5 + 16;
            *(_DWORD *)(v5 + 12) = 0;
            *(_DWORD *)(v5 + 8) = 0;
          }
          else
          {
            while ( v11 != v10 )
            {
              while ( 1 )
              {
                v11 -= 40LL;
                v17 = *(_QWORD *)(v11 + 8);
                if ( v17 == v11 + 24 )
                  break;
                _libc_free(v17);
                if ( v11 == v10 )
                  goto LABEL_23;
              }
            }
LABEL_23:
            *(_DWORD *)(v6 + 48) = 0;
          }
        }
        v13 = *(_BYTE *)(v6 + 80) == 0;
        *(_BYTE *)(v6 + 56) = *(_BYTE *)(v5 + 16);
        if ( !v13 )
          break;
        if ( *(_BYTE *)(v5 + 40) )
        {
          v18 = *(_DWORD *)(v5 + 32);
          v6 += 88;
          v5 += 88;
          *(_DWORD *)(v6 - 16) = v18;
          *(_QWORD *)(v6 - 24) = *(_QWORD *)(v5 - 64);
          LOBYTE(v18) = *(_BYTE *)(v5 - 52);
          *(_DWORD *)(v5 - 56) = 0;
          *(_BYTE *)(v6 - 12) = v18;
          *(_BYTE *)(v6 - 8) = 1;
          if ( !--v4 )
            return v3 + a3;
        }
        else
        {
LABEL_18:
          v6 += 88;
          v5 += 88;
          if ( !--v4 )
            return v3 + a3;
        }
      }
      v14 = *(_DWORD *)(v6 + 72);
      if ( !*(_BYTE *)(v5 + 40) )
      {
        *(_BYTE *)(v6 + 80) = 0;
        if ( v14 > 0x40 )
        {
          v15 = *(_QWORD *)(v6 + 64);
          if ( v15 )
            j_j___libc_free_0_0(v15);
        }
        goto LABEL_18;
      }
      if ( v14 > 0x40 )
      {
        v19 = *(_QWORD *)(v6 + 64);
        if ( v19 )
          j_j___libc_free_0_0(v19);
      }
      v20 = *(_QWORD *)(v5 + 24);
      v6 += 88;
      v5 += 88;
      *(_QWORD *)(v6 - 24) = v20;
      *(_DWORD *)(v6 - 16) = *(_DWORD *)(v5 - 56);
      LOBYTE(v20) = *(_BYTE *)(v5 - 52);
      *(_DWORD *)(v5 - 56) = 0;
      *(_BYTE *)(v6 - 12) = v20;
      if ( !--v4 )
        return v3 + a3;
    }
  }
  return a3;
}
