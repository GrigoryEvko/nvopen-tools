// Function: sub_25FC950
// Address: 0x25fc950
//
__int64 __fastcall sub_25FC950(__int64 a1, __int64 *a2)
{
  __int64 *v3; // rdx
  __int64 *v4; // rbx
  __int64 v5; // rax
  __int64 *v6; // r13
  unsigned int v7; // esi
  __int64 *v8; // rdx
  __int64 v9; // rdi
  __int64 v10; // rax
  __int64 v11; // rdi
  __int64 v12; // r12
  __int64 v13; // r15
  __int64 v14; // r14
  __int64 v15; // rdi
  __int64 v16; // rsi
  __int64 v17; // rax
  __int64 v18; // r10
  __int64 v19; // r11
  __int64 v20; // rax
  __int64 v21; // rsi
  __int64 v22; // rsi
  int v23; // edx
  int v24; // ecx
  __int64 v26; // [rsp+10h] [rbp-70h]
  __int64 v27; // [rsp+18h] [rbp-68h]
  int v28; // [rsp+20h] [rbp-60h]
  char v29; // [rsp+27h] [rbp-59h]
  __int64 v30; // [rsp+28h] [rbp-58h]
  unsigned int v31; // [rsp+38h] [rbp-48h]
  unsigned int v32; // [rsp+3Ch] [rbp-44h]
  __int64 v33; // [rsp+48h] [rbp-38h]

  v28 = 0;
  v27 = *a2;
  v26 = a2[1];
  if ( *a2 == v26 )
  {
LABEL_35:
    BYTE4(v33) = 0;
  }
  else
  {
    while ( *(_DWORD *)(v27 + 16) )
    {
      v3 = *(__int64 **)(v27 + 8);
      v4 = &v3[2 * *(unsigned int *)(v27 + 24)];
      if ( v3 == v4 )
        break;
      while ( 1 )
      {
        v5 = *v3;
        if ( *v3 != -8192 && *v3 != -4096 )
          break;
        v3 += 2;
        if ( v4 == v3 )
          goto LABEL_3;
      }
      v29 = 0;
      if ( v4 == v3 )
        break;
      v6 = v3;
      v32 = *(_DWORD *)(a1 + 24);
      v30 = *(_QWORD *)(a1 + 8);
      v31 = v32 - 1;
      while ( v32 )
      {
        v7 = v31 & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
        v8 = (__int64 *)(v30 + 16LL * v7);
        v9 = *v8;
        if ( *v8 != v5 )
        {
          v23 = 1;
          while ( v9 != -4096 )
          {
            v7 = v31 & (v23 + v7);
            v24 = v23 + 1;
            v8 = (__int64 *)(v30 + 16LL * v7);
            v9 = *v8;
            if ( *v8 == v5 )
              goto LABEL_13;
            v23 = v24;
          }
          break;
        }
LABEL_13:
        if ( (__int64 *)(v30 + 16LL * v32) == v8 )
          break;
        v10 = v6[1];
        v11 = v8[1];
        v12 = *(_QWORD *)(v10 + 56);
        v13 = v10 + 48;
        v14 = *(_QWORD *)(v11 + 56);
        v15 = v11 + 48;
        if ( v10 + 48 == v12 )
        {
          if ( v14 == v15 )
            break;
          v19 = -1;
        }
        else
        {
          v16 = *(_QWORD *)(v10 + 56);
          v17 = 0;
          do
          {
            v16 = *(_QWORD *)(v16 + 8);
            v18 = v17++;
          }
          while ( v13 != v16 );
          v19 = v18;
          if ( v14 == v15 )
          {
            if ( !v18 )
            {
              while ( 1 )
              {
LABEL_22:
                if ( !v12 )
                  BUG();
                if ( *(_BYTE *)(v12 - 24) != 31 )
                {
                  v22 = v14 - 24;
                  if ( !v14 )
                    v22 = 0;
                  if ( !sub_B46220(v12 - 24, v22) )
                  {
                    v29 = 1;
                    goto LABEL_29;
                  }
                  v14 = *(_QWORD *)(v14 + 8);
                }
                v12 = *(_QWORD *)(v12 + 8);
                if ( v13 == v12 )
                  goto LABEL_29;
              }
            }
            break;
          }
        }
        v20 = v14;
        v21 = 0;
        do
        {
          v20 = *(_QWORD *)(v20 + 8);
          ++v21;
        }
        while ( v20 != v15 );
        if ( v19 != v21 )
          break;
        if ( v13 != v12 )
          goto LABEL_22;
LABEL_29:
        v6 += 2;
        if ( v6 != v4 )
        {
          while ( 1 )
          {
            v5 = *v6;
            if ( *v6 != -4096 && v5 != -8192 )
              break;
            v6 += 2;
            if ( v4 == v6 )
              goto LABEL_33;
          }
          if ( v4 != v6 )
            continue;
        }
LABEL_33:
        if ( !v29 )
          goto LABEL_3;
        break;
      }
      v27 += 32;
      ++v28;
      if ( v26 == v27 )
        goto LABEL_35;
    }
LABEL_3:
    BYTE4(v33) = 1;
    LODWORD(v33) = v28;
  }
  return v33;
}
