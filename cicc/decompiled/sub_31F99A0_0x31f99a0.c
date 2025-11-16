// Function: sub_31F99A0
// Address: 0x31f99a0
//
void __fastcall sub_31F99A0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r14
  __int64 v4; // r15
  __int64 v5; // rbx
  char v7; // r13
  __int64 v8; // rdi
  unsigned __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r8
  int v13; // eax
  __int64 v14; // rax
  __int64 v15; // rax
  bool v16; // zf
  unsigned __int64 v17; // rax
  __int64 v18; // [rsp+18h] [rbp-48h]
  __int64 v19; // [rsp+20h] [rbp-40h] BYREF
  _QWORD v20[7]; // [rsp+28h] [rbp-38h] BYREF

  v3 = *(_QWORD *)(a1 + 64);
  if ( v3 )
  {
    if ( *(_QWORD *)(v3 + 8) != *(_QWORD *)(v3 + 16) )
    {
      v4 = *(_QWORD *)(a1 + 328);
      v5 = a1 + 320;
      if ( v4 != a1 + 320 )
      {
        v7 = a2;
        do
        {
          v8 = v4;
          v9 = sub_2E313E0(v4);
          v11 = v4 + 48;
          v12 = v9;
          if ( v9 != v4 + 48 )
          {
            v13 = *(_DWORD *)(v9 + 44);
            if ( (v13 & 4) != 0 || (v13 & 8) == 0 )
            {
              v14 = (*(_QWORD *)(*(_QWORD *)(v12 + 16) + 24LL) >> 11) & 1LL;
            }
            else
            {
              v8 = v12;
              a2 = 2048;
              v18 = v12;
              LOBYTE(v14) = sub_2E88A90(v12, 2048, 1);
              v12 = v18;
              v11 = v4 + 48;
            }
            if ( (_BYTE)v14 )
            {
              if ( v7 )
              {
                v15 = *(_QWORD *)(v12 + 32);
                v10 = v15 + 40LL * (*(_DWORD *)(v12 + 40) & 0xFFFFFF);
                if ( v15 != v10 )
                {
                  while ( *(_BYTE *)v15 != 8 )
                  {
                    v15 += 40;
                    if ( v10 == v15 )
                      goto LABEL_17;
                  }
                  v16 = *(_QWORD *)(a3 + 16) == 0;
                  v19 = *(unsigned int *)(v15 + 24);
                  if ( v16 )
                    goto LABEL_26;
                  a2 = v3;
                  (*(void (__fastcall **)(__int64, __int64, __int64, __int64 *))(a3 + 24))(a3, v3, v12, &v19);
                }
              }
              else
              {
                v17 = *(_QWORD *)(v4 + 48) & 0xFFFFFFFFFFFFFFF8LL;
                if ( v11 != v17 )
                {
                  while ( *(_WORD *)(v17 + 68) != 45 )
                  {
                    v17 = *(_QWORD *)v17 & 0xFFFFFFFFFFFFFFF8LL;
                    if ( v11 == v17 )
                      goto LABEL_17;
                  }
                  v16 = *(_QWORD *)(a3 + 16) == 0;
                  v20[0] = *(unsigned int *)(*(_QWORD *)(v17 + 32) + 24LL);
                  if ( v16 )
LABEL_26:
                    sub_4263D6(v8, a2, v10);
                  a2 = v3;
                  (*(void (__fastcall **)(__int64, __int64, __int64, _QWORD *))(a3 + 24))(a3, v3, v12, v20);
                }
              }
            }
          }
LABEL_17:
          v4 = *(_QWORD *)(v4 + 8);
        }
        while ( v5 != v4 );
      }
    }
  }
}
