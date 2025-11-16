// Function: sub_2E256C0
// Address: 0x2e256c0
//
void __fastcall sub_2E256C0(__int64 a1, __int64 a2)
{
  __int64 v2; // r10
  __int64 v3; // r15
  __int64 v5; // r12
  __int64 v6; // r9
  unsigned int v7; // ebx
  int v8; // r14d
  __int64 v9; // rcx
  __int64 v10; // rax
  int v11; // r11d
  __int64 v12; // r13
  __int64 v13; // rax
  __int64 v14; // [rsp+0h] [rbp-50h]
  int v15; // [rsp+Ch] [rbp-44h]
  __int64 v16; // [rsp+10h] [rbp-40h]
  __int64 v17; // [rsp+18h] [rbp-38h]

  v2 = a2 + 320;
  v3 = *(_QWORD *)(a2 + 328);
  if ( v3 != a2 + 320 )
  {
    do
    {
LABEL_5:
      v5 = *(_QWORD *)(v3 + 56);
      v6 = v3 + 48;
      if ( v5 != v3 + 48 )
      {
        while ( *(_WORD *)(v5 + 68) == 68 || !*(_WORD *)(v5 + 68) )
        {
          v7 = 1;
          v8 = *(_DWORD *)(v5 + 40) & 0xFFFFFF;
          if ( v8 != 1 )
          {
            do
            {
              v9 = *(_QWORD *)(v5 + 32);
              v10 = v9 + 40LL * v7;
              if ( (*(_BYTE *)(v10 + 4) & 1) == 0
                && (*(_BYTE *)(v10 + 4) & 2) == 0
                && ((*(_BYTE *)(v10 + 3) & 0x10) == 0 || (*(_DWORD *)v10 & 0xFFF00) != 0) )
              {
                v11 = *(_DWORD *)(v10 + 8);
                v12 = *(_QWORD *)(a1 + 152) + 32LL * *(int *)(*(_QWORD *)(v9 + 40LL * (v7 + 1) + 24) + 24LL);
                v13 = *(unsigned int *)(v12 + 8);
                if ( v13 + 1 > (unsigned __int64)*(unsigned int *)(v12 + 12) )
                {
                  v14 = a1;
                  v15 = v11;
                  v16 = v2;
                  v17 = v6;
                  sub_C8D5F0(v12, (const void *)(v12 + 16), v13 + 1, 4u, a1, v6);
                  v13 = *(unsigned int *)(v12 + 8);
                  a1 = v14;
                  v11 = v15;
                  v2 = v16;
                  v6 = v17;
                }
                *(_DWORD *)(*(_QWORD *)v12 + 4 * v13) = v11;
                ++*(_DWORD *)(v12 + 8);
              }
              v7 += 2;
            }
            while ( v7 != v8 );
          }
          if ( (*(_BYTE *)v5 & 4) == 0 )
          {
            while ( (*(_BYTE *)(v5 + 44) & 8) != 0 )
              v5 = *(_QWORD *)(v5 + 8);
          }
          v5 = *(_QWORD *)(v5 + 8);
          if ( v6 == v5 )
          {
            v3 = *(_QWORD *)(v3 + 8);
            if ( v2 != v3 )
              goto LABEL_5;
            return;
          }
        }
      }
      v3 = *(_QWORD *)(v3 + 8);
    }
    while ( v2 != v3 );
  }
}
