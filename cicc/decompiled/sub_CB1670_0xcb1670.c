// Function: sub_CB1670
// Address: 0xcb1670
//
void __fastcall sub_CB1670(__int64 a1)
{
  _DWORD *v1; // rcx
  __int64 v2; // rax
  int v3; // ecx
  _QWORD *v4; // rsi
  __int64 *v5; // rax
  __int64 v6; // rdx
  __int64 *v7; // r12
  size_t v8; // r15
  const void *v9; // r13
  __int64 v10; // r14
  __int64 v11; // rbx
  __int64 v12; // rax
  __int64 v13; // rbx
  __int64 v14; // rbx
  __int64 v15; // r8
  int v16; // eax
  int v17; // eax
  int v18; // eax
  unsigned __int64 *v19; // rsi
  bool v20; // zf
  __int64 v21; // rax
  __int64 *v22; // rdx
  __int64 *v23; // rax
  __int64 v24; // rdx
  __int64 *v25; // [rsp+10h] [rbp-C0h]
  _DWORD *v26; // [rsp+18h] [rbp-B8h]
  size_t *v27; // [rsp+28h] [rbp-A8h]
  __int64 v28; // [rsp+30h] [rbp-A0h]
  _QWORD v29[4]; // [rsp+40h] [rbp-90h] BYREF
  __int16 v30; // [rsp+60h] [rbp-70h]
  _QWORD v31[2]; // [rsp+70h] [rbp-60h] BYREF
  char *v32; // [rsp+80h] [rbp-50h]
  __int16 v33; // [rsp+90h] [rbp-40h]

  v1 = *(_DWORD **)(a1 + 672);
  v26 = v1;
  if ( v1 )
  {
    if ( *(_DWORD *)(*(_QWORD *)v1 + 32LL) == 4 )
    {
      v2 = *(_QWORD *)(a1 + 672);
      v3 = v1[4];
      if ( v3 )
      {
        v4 = *(_QWORD **)(v2 + 8);
        if ( *v4 != -8 && *v4 )
        {
          v7 = *(__int64 **)(v2 + 8);
        }
        else
        {
          v5 = v4 + 1;
          do
          {
            do
            {
              v6 = *v5;
              v7 = v5++;
            }
            while ( !v6 );
          }
          while ( v6 == -8 );
        }
        v25 = &v4[v3];
        if ( v25 != v7 )
        {
          while ( 1 )
          {
            v27 = (size_t *)*v7;
            v8 = *(_QWORD *)*v7;
            v9 = (const void *)(*v7 + 32);
            v10 = *((_QWORD *)v26 + 4);
            v11 = 32LL * (unsigned int)v26[10];
            v28 = v10 + v11;
            v12 = v11 >> 5;
            v13 = v11 >> 7;
            if ( v13 )
            {
              v14 = v10 + (v13 << 7);
              while ( v8 != *(_QWORD *)(v10 + 8) || v8 && memcmp(*(const void **)v10, v9, v8) )
              {
                v15 = v10 + 32;
                if ( v8 == *(_QWORD *)(v10 + 40) )
                {
                  if ( !v8 )
                    goto LABEL_35;
                  v16 = memcmp(*(const void **)(v10 + 32), v9, v8);
                  v15 = v10 + 32;
                  if ( !v16 )
                    goto LABEL_35;
                }
                if ( (v15 = v10 + 64, v8 == *(_QWORD *)(v10 + 72))
                  && (!v8 || (v17 = memcmp(*(const void **)(v10 + 64), v9, v8), v15 = v10 + 64, !v17))
                  || (v15 = v10 + 96, v8 == *(_QWORD *)(v10 + 104))
                  && (!v8 || (v18 = memcmp(*(const void **)(v10 + 96), v9, v8), v15 = v10 + 96, !v18)) )
                {
LABEL_35:
                  v10 = v15;
                  break;
                }
                v10 += 128;
                if ( v14 == v10 )
                {
                  v12 = (v28 - v10) >> 5;
                  goto LABEL_37;
                }
              }
LABEL_25:
              if ( v28 != v10 )
                goto LABEL_28;
              goto LABEL_26;
            }
LABEL_37:
            if ( v12 != 2 )
            {
              if ( v12 != 3 )
              {
                if ( v12 != 1 )
                  goto LABEL_26;
                goto LABEL_40;
              }
              if ( v8 == *(_QWORD *)(v10 + 8) && (!v8 || !memcmp(*(const void **)v10, v9, v8)) )
                goto LABEL_25;
              v10 += 32;
            }
            if ( v8 == *(_QWORD *)(v10 + 8) && (!v8 || !memcmp(*(const void **)v10, v9, v8)) )
              goto LABEL_25;
            v10 += 32;
LABEL_40:
            if ( v8 == *(_QWORD *)(v10 + 8) && (!v8 || !memcmp(*(const void **)v10, v9, v8)) )
              goto LABEL_25;
LABEL_26:
            v29[2] = v9;
            v29[3] = v8;
            v19 = v27 + 2;
            v20 = *(_BYTE *)(a1 + 681) == 0;
            v30 = 1283;
            v29[0] = "unknown key '";
            if ( v20 )
            {
              v31[0] = v29;
              v33 = 770;
              v32 = "'";
              sub_CB1630(a1, v19, (__int64)v31);
              return;
            }
            v31[0] = v29;
            v32 = "'";
            v33 = 770;
            sub_CB1660(a1, v19, (__int64)v31);
LABEL_28:
            v21 = v7[1];
            v22 = v7 + 1;
            if ( v21 != -8 && v21 )
            {
              ++v7;
              if ( v22 == v25 )
                return;
            }
            else
            {
              v23 = v7 + 2;
              do
              {
                do
                {
                  v24 = *v23;
                  v7 = v23++;
                }
                while ( !v24 );
              }
              while ( v24 == -8 );
              if ( v7 == v25 )
                return;
            }
          }
        }
      }
    }
  }
}
