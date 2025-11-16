// Function: sub_26EBE40
// Address: 0x26ebe40
//
void __fastcall sub_26EBE40(__int64 *a1, __int64 a2)
{
  __int64 v2; // r14
  __int64 v3; // rax
  __int64 v4; // rbx
  unsigned int v6; // edx
  __int64 *v7; // rdi
  __int64 v8; // r8
  __int64 v9; // rsi
  __int64 v10; // rcx
  __int64 v11; // rax
  __int64 v12; // r13
  int v13; // eax
  unsigned int v14; // ebx
  int v15; // r12d
  unsigned __int64 v16; // rax
  unsigned int v17; // eax
  int v18; // ecx
  unsigned int v19; // r14d
  unsigned int v20; // ebx
  char *v21; // rsi
  unsigned int v22; // ebx
  char *v23; // rsi
  char *v24; // rdx
  unsigned __int64 v25; // rdi
  int v26; // edi
  int v27; // r10d
  __int64 v28; // [rsp+18h] [rbp-88h]
  __int64 v29; // [rsp+28h] [rbp-78h]
  int v30; // [rsp+34h] [rbp-6Ch]
  __int64 v31; // [rsp+38h] [rbp-68h]
  char v32; // [rsp+4Bh] [rbp-55h] BYREF
  unsigned int v33; // [rsp+4Ch] [rbp-54h] BYREF
  unsigned __int64 v34; // [rsp+50h] [rbp-50h] BYREF
  char *v35; // [rsp+58h] [rbp-48h]
  char *v36; // [rsp+60h] [rbp-40h]

  v2 = (__int64)a1;
  v3 = *a1;
  v34 = 0;
  v35 = 0;
  v4 = *(_QWORD *)(v3 + 80);
  v36 = 0;
  v33 = -1;
  v29 = v3 + 72;
  if ( v4 != v3 + 72 )
  {
    while ( 1 )
    {
      v9 = v4 - 24;
      v10 = *(_QWORD *)(a2 + 8);
      if ( !v4 )
        v9 = 0;
      v11 = *(unsigned int *)(a2 + 24);
      if ( !(_DWORD)v11 )
        goto LABEL_9;
      v6 = (v11 - 1) & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
      v7 = (__int64 *)(v10 + 8LL * v6);
      v8 = *v7;
      if ( v9 == *v7 )
      {
LABEL_4:
        if ( v7 == (__int64 *)(v10 + 8 * v11) )
          goto LABEL_9;
LABEL_5:
        v4 = *(_QWORD *)(v4 + 8);
        if ( v29 == v4 )
          goto LABEL_21;
      }
      else
      {
        v26 = 1;
        while ( v8 != -4096 )
        {
          v27 = v26 + 1;
          v6 = (v11 - 1) & (v26 + v6);
          v7 = (__int64 *)(v10 + 8LL * v6);
          v8 = *v7;
          if ( v9 == *v7 )
            goto LABEL_4;
          v26 = v27;
        }
LABEL_9:
        v12 = sub_26E9CC0(v2, v9, a2);
        v13 = sub_B46E30(v12);
        if ( !v13 )
          goto LABEL_5;
        v28 = v4;
        v14 = 0;
        v15 = v13;
        do
        {
          while ( 1 )
          {
            v16 = sub_B46EC0(v12, v14);
            v17 = sub_26EA740(v2, v16);
            if ( v17 )
              break;
            if ( v15 == ++v14 )
              goto LABEL_20;
          }
          v31 = v2;
          v18 = 0;
          v19 = v14;
          v20 = v17;
          do
          {
            v21 = v35;
            v32 = v20 >> v18;
            if ( v35 == v36 )
            {
              v30 = v18;
              sub_C8FB10((__int64)&v34, v35, &v32);
              v18 = v30;
            }
            else
            {
              if ( v35 )
              {
                *v35 = v20 >> v18;
                v21 = v35;
              }
              v35 = v21 + 1;
            }
            v18 += 8;
          }
          while ( v18 != 32 );
          v22 = v19;
          v2 = v31;
          v14 = v22 + 1;
        }
        while ( v15 != v14 );
LABEL_20:
        v4 = *(_QWORD *)(v28 + 8);
        if ( v29 == v4 )
        {
LABEL_21:
          v23 = (char *)v34;
          v24 = &v35[-v34];
          goto LABEL_22;
        }
      }
    }
  }
  v24 = 0;
  v23 = 0;
LABEL_22:
  sub_1098F90(&v33, v23, (__int64)v24);
  v25 = v34;
  *(_QWORD *)(v2 + 40) = ((*(_QWORD *)(v2 + 128) << 48) | v33 | ((_QWORD)&v35[-v34] << 32)) & 0xFFFFFFFFFFFFFFFLL;
  if ( v25 )
    j_j___libc_free_0(v25);
}
