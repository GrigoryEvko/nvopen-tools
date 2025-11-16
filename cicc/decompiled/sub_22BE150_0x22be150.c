// Function: sub_22BE150
// Address: 0x22be150
//
__int64 __fastcall sub_22BE150(__int64 a1, int a2)
{
  unsigned __int64 v2; // rdx
  __int64 v4; // rbx
  __int64 v5; // r12
  unsigned int v6; // eax
  _QWORD *v7; // rax
  __int64 v8; // rdx
  _QWORD *v9; // r13
  _QWORD *v10; // rcx
  char v11; // dl
  _QWORD *v12; // rbx
  __int64 v13; // rax
  __int64 v14; // rdx
  int v15; // eax
  int v16; // ecx
  __int64 v17; // rdi
  unsigned int v18; // eax
  _QWORD *v19; // r15
  __int64 v20; // rsi
  int v22; // r10d
  _QWORD *v23; // r9
  __int64 v24; // rax
  unsigned __int64 *v25; // rdi
  __int64 v26; // [rsp+8h] [rbp-98h]
  _QWORD v27[2]; // [rsp+18h] [rbp-88h] BYREF
  __int64 v28; // [rsp+28h] [rbp-78h]
  __int64 v29; // [rsp+30h] [rbp-70h]
  __int64 (__fastcall **v30)(); // [rsp+40h] [rbp-60h]
  _QWORD v31[2]; // [rsp+48h] [rbp-58h] BYREF
  __int64 v32; // [rsp+58h] [rbp-48h]
  __int64 i; // [rsp+60h] [rbp-40h]

  v2 = (unsigned int)(a2 - 1);
  v4 = *(unsigned int *)(a1 + 24);
  v5 = *(_QWORD *)(a1 + 8);
  v6 = (((((((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4) | ((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 8)
        | ((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4)
        | ((v2 | (v2 >> 1)) >> 2)
        | v2
        | (v2 >> 1)) >> 16)
      | ((((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4) | ((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 8)
      | ((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4)
      | ((v2 | (v2 >> 1)) >> 2)
      | v2
      | (v2 >> 1))
     + 1;
  if ( v6 < 0x40 )
    v6 = 64;
  *(_DWORD *)(a1 + 24) = v6;
  v7 = (_QWORD *)sub_C7D670(40LL * v6, 8);
  *(_QWORD *)(a1 + 8) = v7;
  if ( !v5 )
    return sub_22BDDA0(a1);
  v8 = *(unsigned int *)(a1 + 24);
  *(_QWORD *)(a1 + 16) = 0;
  v31[0] = 2;
  v26 = 40 * v4;
  v9 = (_QWORD *)(v5 + 40 * v4);
  v10 = &v7[5 * v8];
  for ( i = 0; v10 != v7; v7 += 5 )
  {
    if ( v7 )
    {
      v11 = v31[0];
      v7[2] = 0;
      v7[3] = -4096;
      *v7 = off_4A09D90;
      v7[1] = v11 & 6;
      v7[4] = i;
    }
  }
  v27[0] = 2;
  v12 = (_QWORD *)v5;
  v13 = -4096;
  v27[1] = 0;
  v28 = -4096;
  v29 = 0;
  v31[0] = 2;
  v31[1] = 0;
  v32 = -8192;
  v30 = off_4A09D90;
  i = 0;
  if ( v9 != (_QWORD *)v5 )
  {
    while ( 1 )
    {
      v14 = v12[3];
      if ( v14 != v13 )
      {
        v13 = v32;
        if ( v14 != v32 )
        {
          v15 = *(_DWORD *)(a1 + 24);
          if ( !v15 )
            BUG();
          v16 = v15 - 1;
          v17 = *(_QWORD *)(a1 + 8);
          v18 = (v15 - 1) & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
          v19 = (_QWORD *)(v17 + 40LL * v18);
          v20 = v19[3];
          if ( v14 != v20 )
          {
            v22 = 1;
            v23 = 0;
            while ( v20 != -4096 )
            {
              if ( v20 == -8192 && !v23 )
                v23 = v19;
              v18 = v16 & (v22 + v18);
              v19 = (_QWORD *)(v17 + 40LL * v18);
              v20 = v19[3];
              if ( v14 == v20 )
                goto LABEL_15;
              ++v22;
            }
            if ( v23 )
            {
              v24 = v23[3];
              v19 = v23;
            }
            else
            {
              v24 = v19[3];
            }
            v25 = v19 + 1;
            if ( v14 != v24 )
            {
              if ( v24 != 0 && v24 != -4096 && v24 != -8192 )
              {
                sub_BD60C0(v25);
                v14 = v12[3];
                v25 = v19 + 1;
              }
              v19[3] = v14;
              if ( v14 != -4096 && v14 != 0 && v14 != -8192 )
                sub_BD6050(v25, v12[1] & 0xFFFFFFFFFFFFFFF8LL);
            }
          }
LABEL_15:
          v19[4] = v12[4];
          ++*(_DWORD *)(a1 + 16);
          v13 = v12[3];
        }
      }
      *v12 = &unk_49DB368;
      if ( v13 != 0 && v13 != -4096 && v13 != -8192 )
        sub_BD60C0(v12 + 1);
      v12 += 5;
      if ( v9 == v12 )
        break;
      v13 = v28;
    }
    v30 = (__int64 (__fastcall **)())&unk_49DB368;
    if ( v32 != 0 && v32 != -8192 && v32 != -4096 )
      sub_BD60C0(v31);
  }
  if ( v28 != -4096 && v28 != 0 && v28 != -8192 )
    sub_BD60C0(v27);
  return sub_C7D6A0(v5, v26, 8);
}
