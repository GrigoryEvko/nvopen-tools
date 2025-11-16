// Function: sub_2631930
// Address: 0x2631930
//
void __fastcall sub_2631930(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r12
  const void *v8; // r14
  _BYTE *v9; // r13
  int v10; // eax
  __int64 v11; // rcx
  _QWORD *v12; // rax
  _QWORD *v13; // rsi
  __int64 v14; // rdi
  _QWORD *v15; // rdx
  unsigned int v16; // esi
  int v17; // r11d
  _QWORD *v18; // r10
  unsigned int v19; // edx
  _QWORD *v20; // rcx
  _BYTE *v21; // rdi
  int v22; // eax
  __int64 v23; // rax
  __int64 v24; // r13
  __int64 v25; // rax
  __int64 *v26; // r13
  __int64 *v27; // r15
  unsigned int v28; // eax
  __int64 *v29; // rdi
  __int64 v30; // rcx
  unsigned int v31; // esi
  __int64 *v32; // r10
  int v33; // edx
  int v34; // r11d
  int v35; // eax
  _BYTE *v36; // [rsp+10h] [rbp-40h] BYREF
  _QWORD v37[7]; // [rsp+18h] [rbp-38h] BYREF

  v6 = *(_QWORD *)(a1 + 16);
  if ( v6 )
  {
    v8 = (const void *)(a2 + 48);
    do
    {
      v9 = *(_BYTE **)(v6 + 24);
      if ( *v9 != 3 )
      {
        v36 = 0;
        if ( *v9 <= 0x15u )
          sub_2631930(v9, a2);
        goto LABEL_13;
      }
      v10 = *(_DWORD *)(a2 + 16);
      v36 = *(_BYTE **)(v6 + 24);
      if ( v10 )
      {
        v16 = *(_DWORD *)(a2 + 24);
        if ( !v16 )
        {
          ++*(_QWORD *)a2;
          v37[0] = 0;
          goto LABEL_62;
        }
        a6 = v16 - 1;
        a5 = *(_QWORD *)(a2 + 8);
        v17 = 1;
        v18 = 0;
        v19 = a6 & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
        v20 = (_QWORD *)(a5 + 8LL * v19);
        v21 = (_BYTE *)*v20;
        if ( v9 != (_BYTE *)*v20 )
        {
          while ( v21 != (_BYTE *)-4096LL )
          {
            if ( v21 != (_BYTE *)-8192LL || v18 )
              v20 = v18;
            v19 = a6 & (v17 + v19);
            v21 = *(_BYTE **)(a5 + 8LL * v19);
            if ( v9 == v21 )
              goto LABEL_13;
            ++v17;
            v18 = v20;
            v20 = (_QWORD *)(a5 + 8LL * v19);
          }
          if ( !v18 )
            v18 = v20;
          v22 = v10 + 1;
          ++*(_QWORD *)a2;
          v37[0] = v18;
          if ( 4 * v22 < 3 * v16 )
          {
            if ( v16 - *(_DWORD *)(a2 + 20) - v22 > v16 >> 3 )
            {
LABEL_24:
              *(_DWORD *)(a2 + 16) = v22;
              if ( *v18 != -4096 )
                --*(_DWORD *)(a2 + 20);
              *v18 = v9;
              v23 = *(unsigned int *)(a2 + 40);
              v24 = (__int64)v36;
              if ( v23 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 44) )
              {
                sub_C8D5F0(a2 + 32, v8, v23 + 1, 8u, a5, a6);
                v23 = *(unsigned int *)(a2 + 40);
              }
              *(_QWORD *)(*(_QWORD *)(a2 + 32) + 8 * v23) = v24;
              ++*(_DWORD *)(a2 + 40);
              goto LABEL_13;
            }
LABEL_63:
            sub_2631760(a2, v16);
            sub_262CE20(a2, (__int64 *)&v36, v37);
            v9 = v36;
            v18 = (_QWORD *)v37[0];
            v22 = *(_DWORD *)(a2 + 16) + 1;
            goto LABEL_24;
          }
LABEL_62:
          v16 *= 2;
          goto LABEL_63;
        }
      }
      else
      {
        v11 = *(unsigned int *)(a2 + 40);
        v12 = *(_QWORD **)(a2 + 32);
        v13 = &v12[v11];
        v14 = (8 * v11) >> 3;
        if ( !((8 * v11) >> 5) )
          goto LABEL_30;
        v15 = &v12[4 * ((8 * v11) >> 5)];
        do
        {
          if ( v9 == (_BYTE *)*v12 )
            goto LABEL_12;
          if ( v9 == (_BYTE *)v12[1] )
          {
            ++v12;
            goto LABEL_12;
          }
          if ( v9 == (_BYTE *)v12[2] )
          {
            v12 += 2;
            goto LABEL_12;
          }
          if ( v9 == (_BYTE *)v12[3] )
          {
            v12 += 3;
            goto LABEL_12;
          }
          v12 += 4;
        }
        while ( v12 != v15 );
        v14 = v13 - v12;
LABEL_30:
        if ( v14 == 2 )
          goto LABEL_48;
        if ( v14 == 3 )
        {
          if ( v9 == (_BYTE *)*v12 )
            goto LABEL_12;
          ++v12;
LABEL_48:
          if ( v9 != (_BYTE *)*v12 )
          {
            ++v12;
            goto LABEL_50;
          }
LABEL_12:
          if ( v13 == v12 )
            goto LABEL_33;
          goto LABEL_13;
        }
        if ( v14 != 1 )
          goto LABEL_33;
LABEL_50:
        if ( v9 == (_BYTE *)*v12 )
          goto LABEL_12;
LABEL_33:
        if ( v11 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 44) )
        {
          sub_C8D5F0(a2 + 32, v8, v11 + 1, 8u, a5, a6);
          v13 = (_QWORD *)(*(_QWORD *)(a2 + 32) + 8LL * *(unsigned int *)(a2 + 40));
        }
        *v13 = v9;
        v25 = (unsigned int)(*(_DWORD *)(a2 + 40) + 1);
        *(_DWORD *)(a2 + 40) = v25;
        if ( (unsigned int)v25 > 8 )
        {
          v26 = *(__int64 **)(a2 + 32);
          v27 = &v26[v25];
          while ( 1 )
          {
            v31 = *(_DWORD *)(a2 + 24);
            if ( !v31 )
              break;
            a6 = v31 - 1;
            a5 = *(_QWORD *)(a2 + 8);
            v28 = a6 & (((unsigned int)*v26 >> 9) ^ ((unsigned int)*v26 >> 4));
            v29 = (__int64 *)(a5 + 8LL * v28);
            v30 = *v29;
            if ( *v26 != *v29 )
            {
              v34 = 1;
              v32 = 0;
              while ( v30 != -4096 )
              {
                if ( v32 || v30 != -8192 )
                  v29 = v32;
                v28 = a6 & (v34 + v28);
                v30 = *(_QWORD *)(a5 + 8LL * v28);
                if ( *v26 == v30 )
                  goto LABEL_38;
                ++v34;
                v32 = v29;
                v29 = (__int64 *)(a5 + 8LL * v28);
              }
              v35 = *(_DWORD *)(a2 + 16);
              if ( !v32 )
                v32 = v29;
              ++*(_QWORD *)a2;
              v33 = v35 + 1;
              v37[0] = v32;
              if ( 4 * (v35 + 1) < 3 * v31 )
              {
                if ( v31 - *(_DWORD *)(a2 + 20) - v33 > v31 >> 3 )
                  goto LABEL_58;
                goto LABEL_42;
              }
LABEL_41:
              v31 *= 2;
LABEL_42:
              sub_2631760(a2, v31);
              sub_262CE20(a2, v26, v37);
              v32 = (__int64 *)v37[0];
              v33 = *(_DWORD *)(a2 + 16) + 1;
LABEL_58:
              *(_DWORD *)(a2 + 16) = v33;
              if ( *v32 != -4096 )
                --*(_DWORD *)(a2 + 20);
              *v32 = *v26;
            }
LABEL_38:
            if ( v27 == ++v26 )
              goto LABEL_13;
          }
          ++*(_QWORD *)a2;
          v37[0] = 0;
          goto LABEL_41;
        }
      }
LABEL_13:
      v6 = *(_QWORD *)(v6 + 8);
    }
    while ( v6 );
  }
}
