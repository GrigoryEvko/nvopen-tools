// Function: sub_29E0880
// Address: 0x29e0880
//
__int64 __fastcall sub_29E0880(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 result; // rax
  __int64 v5; // r15
  __int64 v6; // r14
  __int64 v7; // rbx
  __int64 v8; // rax
  int v9; // edi
  __int64 v10; // r8
  int v11; // edi
  unsigned int v12; // esi
  __int64 *v13; // rdx
  __int64 v14; // r10
  __int64 v15; // rsi
  __int64 v16; // rax
  int v17; // edi
  __int64 v18; // r8
  int v19; // edi
  unsigned int v20; // esi
  __int64 *v21; // rdx
  __int64 v22; // r10
  __int64 v23; // rsi
  __int64 v24; // rax
  int v25; // esi
  __int64 v26; // r8
  __int64 v27; // rdi
  int v28; // esi
  unsigned int v29; // edx
  __int64 *v30; // rax
  __int64 v31; // r10
  __int64 v32; // rsi
  __int64 *v33; // rax
  __int64 v34; // rax
  __int64 v35; // rbx
  __int64 v36; // rdx
  __int64 v37; // rdx
  int v38; // edx
  int v39; // r9d
  int v40; // edx
  int v41; // r9d
  int v42; // eax
  int v43; // r9d
  __int64 v45; // [rsp+10h] [rbp-50h]
  __int64 v46; // [rsp+18h] [rbp-48h]
  __int64 v47; // [rsp+18h] [rbp-48h]
  _BYTE *v48; // [rsp+18h] [rbp-48h]
  __int64 v49[7]; // [rsp+28h] [rbp-38h] BYREF

  result = *(unsigned int *)(a1 + 64);
  if ( (_DWORD)result )
  {
    v45 = a2;
    if ( a2 != a3 )
    {
      while ( 1 )
      {
        if ( !v45 )
          BUG();
        v5 = *(_QWORD *)(v45 + 32);
        v6 = v45 + 24;
        if ( v5 != v45 + 24 )
          break;
LABEL_39:
        result = *(_QWORD *)(v45 + 8);
        v45 = result;
        if ( result == a3 )
          return result;
      }
      while ( 1 )
      {
        if ( !v5 )
          BUG();
        v7 = v5 - 24;
        if ( (*(_BYTE *)(v5 - 17) & 0x20) != 0 )
        {
          v8 = sub_B91C10(v5 - 24, 7);
          if ( v8 )
          {
            v9 = *(_DWORD *)(a1 + 72);
            v10 = *(_QWORD *)(a1 + 56);
            if ( v9 )
            {
              v11 = v9 - 1;
              v12 = v11 & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
              v13 = (__int64 *)(v10 + 16LL * v12);
              v14 = *v13;
              if ( v8 == *v13 )
              {
LABEL_12:
                v15 = v13[1];
                v49[0] = v15;
                if ( v15 )
                {
                  sub_B96E90((__int64)v49, v15, 1);
                  if ( v49[0] )
                  {
                    v46 = v49[0];
                    sub_B91220((__int64)v49, v49[0]);
                    sub_B99FD0(v5 - 24, 7u, v46);
                  }
                }
              }
              else
              {
                v40 = 1;
                while ( v14 != -4096 )
                {
                  v41 = v40 + 1;
                  v12 = v11 & (v40 + v12);
                  v13 = (__int64 *)(v10 + 16LL * v12);
                  v14 = *v13;
                  if ( v8 == *v13 )
                    goto LABEL_12;
                  v40 = v41;
                }
              }
            }
          }
          if ( (*(_BYTE *)(v5 - 17) & 0x20) != 0 )
          {
            v16 = sub_B91C10(v5 - 24, 8);
            if ( v16 )
            {
              v17 = *(_DWORD *)(a1 + 72);
              v18 = *(_QWORD *)(a1 + 56);
              if ( v17 )
              {
                v19 = v17 - 1;
                v20 = v19 & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
                v21 = (__int64 *)(v18 + 16LL * v20);
                v22 = *v21;
                if ( v16 == *v21 )
                {
LABEL_19:
                  v23 = v21[1];
                  v49[0] = v23;
                  if ( v23 )
                  {
                    sub_B96E90((__int64)v49, v23, 1);
                    if ( v49[0] )
                    {
                      v47 = v49[0];
                      sub_B91220((__int64)v49, v49[0]);
                      sub_B99FD0(v5 - 24, 8u, v47);
                    }
                  }
                }
                else
                {
                  v38 = 1;
                  while ( v22 != -4096 )
                  {
                    v39 = v38 + 1;
                    v20 = v19 & (v38 + v20);
                    v21 = (__int64 *)(v18 + 16LL * v20);
                    v22 = *v21;
                    if ( v16 == *v21 )
                      goto LABEL_19;
                    v38 = v39;
                  }
                }
              }
            }
          }
        }
        if ( *(_BYTE *)(v5 - 24) != 85 )
          goto LABEL_6;
        v24 = *(_QWORD *)(v5 - 56);
        if ( !v24 )
          goto LABEL_6;
        if ( *(_BYTE *)v24 )
          goto LABEL_6;
        if ( *(_QWORD *)(v24 + 24) != *(_QWORD *)(v5 + 56) )
          goto LABEL_6;
        if ( (*(_BYTE *)(v24 + 33) & 0x20) == 0 )
          goto LABEL_6;
        if ( *(_DWORD *)(v24 + 36) != 155 )
          goto LABEL_6;
        v25 = *(_DWORD *)(a1 + 72);
        v26 = *(_QWORD *)(a1 + 56);
        v27 = *(_QWORD *)(*(_QWORD *)(v7 - 32LL * (*(_DWORD *)(v5 - 20) & 0x7FFFFFF)) + 24LL);
        if ( !v25 )
          goto LABEL_6;
        v28 = v25 - 1;
        v29 = v28 & (((unsigned int)v27 >> 9) ^ ((unsigned int)v27 >> 4));
        v30 = (__int64 *)(v26 + 16LL * v29);
        v31 = *v30;
        if ( v27 != *v30 )
          break;
LABEL_30:
        v32 = v30[1];
        v49[0] = v32;
        if ( !v32 )
          goto LABEL_6;
        sub_B96E90((__int64)v49, v32, 1);
        if ( !v49[0] )
          goto LABEL_6;
        v48 = (_BYTE *)v49[0];
        sub_B91220((__int64)v49, v49[0]);
        v33 = (__int64 *)sub_BD5C60(v5 - 24);
        v34 = sub_B9F6F0(v33, v48);
        v35 = v7 - 32LL * (*(_DWORD *)(v5 - 20) & 0x7FFFFFF);
        if ( *(_QWORD *)v35 )
        {
          v36 = *(_QWORD *)(v35 + 8);
          **(_QWORD **)(v35 + 16) = v36;
          if ( v36 )
            *(_QWORD *)(v36 + 16) = *(_QWORD *)(v35 + 16);
        }
        *(_QWORD *)v35 = v34;
        if ( v34 )
        {
          v37 = *(_QWORD *)(v34 + 16);
          *(_QWORD *)(v35 + 8) = v37;
          if ( v37 )
            *(_QWORD *)(v37 + 16) = v35 + 8;
          *(_QWORD *)(v35 + 16) = v34 + 16;
          *(_QWORD *)(v34 + 16) = v35;
          v5 = *(_QWORD *)(v5 + 8);
          if ( v6 == v5 )
            goto LABEL_39;
        }
        else
        {
LABEL_6:
          v5 = *(_QWORD *)(v5 + 8);
          if ( v6 == v5 )
            goto LABEL_39;
        }
      }
      v42 = 1;
      while ( v31 != -4096 )
      {
        v43 = v42 + 1;
        v29 = v28 & (v42 + v29);
        v30 = (__int64 *)(v26 + 16LL * v29);
        v31 = *v30;
        if ( v27 == *v30 )
          goto LABEL_30;
        v42 = v43;
      }
      goto LABEL_6;
    }
  }
  return result;
}
