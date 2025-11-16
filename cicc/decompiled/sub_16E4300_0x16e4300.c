// Function: sub_16E4300
// Address: 0x16e4300
//
void __fastcall sub_16E4300(__int64 a1)
{
  __int64 v1; // rcx
  int v2; // esi
  _QWORD *v3; // rdi
  __int64 *v4; // rax
  __int64 v5; // rdx
  size_t **v6; // r14
  size_t **v7; // r15
  __int64 v8; // rax
  __int64 v9; // rbx
  size_t v10; // r14
  const void *v11; // r12
  __int64 v12; // r13
  const void *v13; // rdi
  const void *v14; // rdi
  const void *v15; // rdi
  size_t *v16; // rax
  __int64 v17; // r9
  int v18; // eax
  int v19; // eax
  int v20; // eax
  __int64 v21; // rsi
  __int64 v22; // rax
  __int64 v23; // [rsp+0h] [rbp-C0h]
  size_t **v25; // [rsp+10h] [rbp-B0h]
  size_t *v26; // [rsp+18h] [rbp-A8h]
  __int64 v27; // [rsp+20h] [rbp-A0h]
  __int64 v28; // [rsp+28h] [rbp-98h]
  __int64 v29; // [rsp+30h] [rbp-90h]
  __int64 v30; // [rsp+38h] [rbp-88h]
  __int64 v31; // [rsp+38h] [rbp-88h]
  __int64 v32; // [rsp+38h] [rbp-88h]
  _QWORD v33[2]; // [rsp+40h] [rbp-80h] BYREF
  _QWORD v34[2]; // [rsp+50h] [rbp-70h] BYREF
  __int16 v35; // [rsp+60h] [rbp-60h]
  _QWORD *v36; // [rsp+70h] [rbp-50h]
  char *v37; // [rsp+78h] [rbp-48h]
  __int16 v38; // [rsp+80h] [rbp-40h]

  v1 = *(_QWORD *)(a1 + 264);
  if ( v1 )
  {
    if ( *(_DWORD *)(*(_QWORD *)(v1 + 8) + 32LL) == 4 )
    {
      v2 = *(_DWORD *)(v1 + 24);
      if ( v2 )
      {
        v3 = *(_QWORD **)(v1 + 16);
        if ( *v3 && *v3 != -8 )
        {
          v6 = *(size_t ***)(v1 + 16);
        }
        else
        {
          v4 = v3 + 1;
          do
          {
            do
            {
              v5 = *v4;
              v6 = (size_t **)v4++;
            }
            while ( !v5 );
          }
          while ( v5 == -8 );
        }
        v25 = (size_t **)&v3[v2];
        if ( v25 != v6 )
        {
          v7 = v6;
          v8 = 32LL * *(unsigned int *)(v1 + 56);
          v27 = *(_QWORD *)(v1 + 48);
          v29 = v27 + v8;
          v28 = v8 >> 7;
          v23 = v8 >> 5;
          while ( 1 )
          {
            v9 = v28;
            v26 = *v7;
            v10 = **v7;
            v11 = *v7 + 2;
            if ( v28 )
            {
              v12 = v27;
              while ( v10 != *(_QWORD *)(v12 + 8) )
              {
                v13 = *(const void **)(v12 + 32);
                if ( v10 == *(_QWORD *)(v12 + 40) )
                {
                  v17 = v12 + 32;
                  if ( !v10 )
                    goto LABEL_33;
                  goto LABEL_27;
                }
                v14 = *(const void **)(v12 + 64);
                if ( v10 == *(_QWORD *)(v12 + 72) )
                {
                  v17 = v12 + 64;
                  if ( !v10 )
                  {
LABEL_33:
                    v12 = v17;
                    goto LABEL_20;
                  }
LABEL_30:
                  v31 = v17;
                  v19 = memcmp(v14, v11, v10);
                  v17 = v31;
                  if ( !v19 )
                    goto LABEL_33;
                  v15 = *(const void **)(v12 + 96);
                  v17 = v12 + 96;
                  if ( v10 == *(_QWORD *)(v12 + 104) )
                    goto LABEL_32;
                  goto LABEL_16;
                }
LABEL_15:
                v15 = *(const void **)(v12 + 96);
                if ( v10 == *(_QWORD *)(v12 + 104) )
                {
                  v17 = v12 + 96;
                  if ( !v10 )
                    goto LABEL_33;
LABEL_32:
                  v32 = v17;
                  v20 = memcmp(v15, v11, v10);
                  v17 = v32;
                  if ( !v20 )
                    goto LABEL_33;
                }
LABEL_16:
                v12 += 128;
                if ( !--v9 )
                {
                  v22 = (v29 - v12) >> 5;
                  goto LABEL_44;
                }
              }
              if ( !v10 || !memcmp(*(const void **)v12, v11, v10) )
                goto LABEL_20;
              v13 = *(const void **)(v12 + 32);
              v17 = v12 + 32;
              if ( v10 == *(_QWORD *)(v12 + 40) )
              {
LABEL_27:
                v30 = v17;
                v18 = memcmp(v13, v11, v10);
                v17 = v30;
                if ( !v18 )
                  goto LABEL_33;
              }
              v14 = *(const void **)(v12 + 64);
              if ( v10 == *(_QWORD *)(v12 + 72) )
              {
                v17 = v12 + 64;
                goto LABEL_30;
              }
              goto LABEL_15;
            }
            v22 = v23;
            v12 = v27;
LABEL_44:
            if ( v22 == 2 )
              goto LABEL_48;
            if ( v22 == 3 )
              break;
            if ( v22 != 1 || v10 != *(_QWORD *)(v12 + 8) )
            {
LABEL_42:
              v33[1] = v10;
              v34[0] = "unknown key '";
              v34[1] = v33;
              v35 = 1283;
              v36 = v34;
              v37 = "'";
              v38 = 770;
              v21 = v26[1];
              v33[0] = v11;
              sub_16E42A0(a1, v21);
              return;
            }
LABEL_50:
            if ( v10 && memcmp(*(const void **)v12, v11, v10) )
              goto LABEL_42;
LABEL_20:
            if ( v29 == v12 )
              goto LABEL_42;
            v16 = v7[1];
            for ( ++v7; v16 == (size_t *)-8LL; ++v7 )
LABEL_22:
              v16 = v7[1];
            if ( !v16 )
              goto LABEL_22;
            if ( v7 == v25 )
              return;
          }
          if ( v10 == *(_QWORD *)(v12 + 8) && (!v10 || !memcmp(*(const void **)v12, v11, v10)) )
            goto LABEL_20;
          v12 += 32;
LABEL_48:
          if ( v10 == *(_QWORD *)(v12 + 8) && (!v10 || !memcmp(*(const void **)v12, v11, v10)) )
            goto LABEL_20;
          v12 += 32;
          if ( v10 != *(_QWORD *)(v12 + 8) )
            goto LABEL_42;
          goto LABEL_50;
        }
      }
    }
  }
}
