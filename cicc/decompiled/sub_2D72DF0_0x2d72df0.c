// Function: sub_2D72DF0
// Address: 0x2d72df0
//
void *__fastcall sub_2D72DF0(__int64 a1, int a2)
{
  unsigned __int64 v2; // rax
  __int64 v4; // r14
  __int64 v5; // r13
  unsigned int v6; // edi
  _QWORD *v7; // rax
  __int64 v8; // rcx
  _QWORD *v9; // r12
  _QWORD *v10; // rcx
  char v11; // dl
  _QWORD *v12; // rbx
  __int64 v13; // rax
  __int64 v14; // rdx
  int v15; // eax
  __int64 v16; // rdi
  unsigned int v17; // ecx
  _QWORD *v18; // r14
  __int64 v19; // r8
  __int64 v20; // rdx
  __int64 v21; // rdx
  __int64 v22; // rax
  int v24; // r10d
  _QWORD *v25; // r9
  __int64 v26; // rcx
  unsigned __int64 *v27; // r8
  __int64 v28; // [rsp+8h] [rbp-98h]
  _QWORD v29[2]; // [rsp+18h] [rbp-88h] BYREF
  __int64 v30; // [rsp+28h] [rbp-78h]
  __int64 v31; // [rsp+30h] [rbp-70h]
  void *v32; // [rsp+40h] [rbp-60h]
  _QWORD v33[2]; // [rsp+48h] [rbp-58h] BYREF
  __int64 v34; // [rsp+58h] [rbp-48h]
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
  v7 = (_QWORD *)sub_C7D670((unsigned __int64)v6 << 6, 8);
  *(_QWORD *)(a1 + 8) = v7;
  if ( !v5 )
    return sub_2D69A40(a1);
  v8 = *(unsigned int *)(a1 + 24);
  *(_QWORD *)(a1 + 16) = 0;
  v28 = v4 << 6;
  v9 = (_QWORD *)(v5 + (v4 << 6));
  v33[0] = 2;
  v10 = &v7[8 * v8];
  for ( i = 0; v10 != v7; v7 += 8 )
  {
    if ( v7 )
    {
      v11 = v33[0];
      v7[2] = 0;
      v7[3] = -4096;
      *v7 = &unk_4A26638;
      v7[1] = v11 & 6;
      v7[4] = i;
    }
  }
  v29[1] = 0;
  v12 = (_QWORD *)v5;
  v32 = &unk_4A26638;
  v13 = -4096;
  v29[0] = 2;
  v30 = -4096;
  v31 = 0;
  v33[0] = 2;
  v33[1] = 0;
  v34 = -8192;
  i = 0;
  if ( v9 != (_QWORD *)v5 )
  {
    while ( 1 )
    {
      v14 = v12[3];
      if ( v14 != v13 )
      {
        v13 = v34;
        if ( v14 != v34 )
        {
          v15 = *(_DWORD *)(a1 + 24);
          if ( !v15 )
            BUG();
          v16 = *(_QWORD *)(a1 + 8);
          v17 = (v15 - 1) & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
          v18 = (_QWORD *)(v16 + ((unsigned __int64)v17 << 6));
          v19 = v18[3];
          if ( v14 != v19 )
          {
            v24 = 1;
            v25 = 0;
            while ( v19 != -4096 )
            {
              if ( v19 == -8192 && !v25 )
                v25 = v18;
              v17 = (v15 - 1) & (v24 + v17);
              v18 = (_QWORD *)(v16 + ((unsigned __int64)v17 << 6));
              v19 = v18[3];
              if ( v14 == v19 )
                goto LABEL_15;
              ++v24;
            }
            if ( v25 )
            {
              v26 = v25[3];
              v18 = v25;
            }
            else
            {
              v26 = v18[3];
            }
            v27 = v18 + 1;
            if ( v14 != v26 )
            {
              if ( v26 != 0 && v26 != -4096 && v26 != -8192 )
              {
                sub_BD60C0(v18 + 1);
                v14 = v12[3];
                v27 = v18 + 1;
              }
              v18[3] = v14;
              if ( v14 != 0 && v14 != -4096 && v14 != -8192 )
                sub_BD6050(v27, v12[1] & 0xFFFFFFFFFFFFFFF8LL);
            }
          }
LABEL_15:
          v20 = v12[4];
          v18[5] = 6;
          v18[6] = 0;
          v18[4] = v20;
          v21 = v12[7];
          v18[7] = v21;
          if ( v21 != -4096 && v21 != 0 && v21 != -8192 )
            sub_BD6050(v18 + 5, v12[5] & 0xFFFFFFFFFFFFFFF8LL);
          ++*(_DWORD *)(a1 + 16);
          v22 = v12[7];
          if ( v22 != 0 && v22 != -4096 && v22 != -8192 )
            sub_BD60C0(v12 + 5);
          v13 = v12[3];
        }
      }
      *v12 = &unk_49DB368;
      if ( v13 != -4096 && v13 != 0 && v13 != -8192 )
        sub_BD60C0(v12 + 1);
      v12 += 8;
      if ( v9 == v12 )
        break;
      v13 = v30;
    }
    v32 = &unk_49DB368;
    if ( v34 != -8192 && v34 != -4096 && v34 )
      sub_BD60C0(v33);
  }
  if ( v30 != 0 && v30 != -4096 && v30 != -8192 )
    sub_BD60C0(v29);
  return (void *)sub_C7D6A0(v5, v28, 8);
}
