// Function: sub_3435920
// Address: 0x3435920
//
__int64 *__fastcall sub_3435920(__int64 a1, __int64 a2)
{
  int *v4; // rsi
  __int64 *result; // rax
  __int64 v6; // rax
  unsigned int v7; // esi
  __int64 v8; // r14
  int v9; // edx
  int v10; // ecx
  __int64 v11; // r9
  int v12; // r11d
  __int64 *v13; // rdx
  unsigned int v14; // r8d
  __int64 *v15; // rax
  __int64 v16; // rdi
  __int64 v17; // r12
  int v18; // edx
  int v19; // r14d
  int v20; // eax
  int v21; // edi
  int v22; // eax
  int v23; // esi
  __int64 v24; // r9
  unsigned int v25; // eax
  __int64 v26; // r8
  int v27; // r11d
  __int64 *v28; // r10
  int v29; // eax
  int v30; // eax
  __int64 v31; // r8
  __int64 *v32; // r9
  unsigned int v33; // r15d
  int v34; // r10d
  __int64 v35; // rsi
  int v36; // [rsp+8h] [rbp-68h]
  int v37; // [rsp+8h] [rbp-68h]
  __int64 v38[7]; // [rsp+38h] [rbp-38h] BYREF

  v4 = (int *)sub_B5B6B0(a2);
  result = (__int64 *)((unsigned int)*(unsigned __int8 *)v4 - 12);
  if ( (unsigned int)result > 1 )
  {
    if ( *(_QWORD *)(a2 + 40) != *((_QWORD *)v4 + 5) )
    {
      v6 = sub_3380740(a1, v4, *(_QWORD *)(a2 + 8));
      v7 = *(_DWORD *)(a1 + 32);
      v8 = v6;
      v10 = v9;
      if ( v7 )
      {
        v11 = *(_QWORD *)(a1 + 16);
        v12 = 1;
        v13 = 0;
        v14 = (v7 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
        v15 = (__int64 *)(v11 + 24LL * v14);
        v16 = *v15;
        if ( a2 == *v15 )
        {
LABEL_5:
          result = v15 + 1;
LABEL_6:
          *result = v8;
          *((_DWORD *)result + 2) = v10;
          return result;
        }
        while ( v16 != -4096 )
        {
          if ( !v13 && v16 == -8192 )
            v13 = v15;
          v14 = (v7 - 1) & (v12 + v14);
          v15 = (__int64 *)(v11 + 24LL * v14);
          v16 = *v15;
          if ( a2 == *v15 )
            goto LABEL_5;
          ++v12;
        }
        if ( !v13 )
          v13 = v15;
        v20 = *(_DWORD *)(a1 + 24);
        ++*(_QWORD *)(a1 + 8);
        v21 = v20 + 1;
        if ( 4 * (v20 + 1) < 3 * v7 )
        {
          if ( v7 - *(_DWORD *)(a1 + 28) - v21 > v7 >> 3 )
          {
LABEL_18:
            *(_DWORD *)(a1 + 24) = v21;
            if ( *v13 != -4096 )
              --*(_DWORD *)(a1 + 28);
            *v13 = a2;
            result = v13 + 1;
            v13[1] = 0;
            *((_DWORD *)v13 + 4) = 0;
            goto LABEL_6;
          }
          v37 = v10;
          sub_337DA20(a1 + 8, v7);
          v29 = *(_DWORD *)(a1 + 32);
          if ( v29 )
          {
            v30 = v29 - 1;
            v31 = *(_QWORD *)(a1 + 16);
            v32 = 0;
            v33 = v30 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
            v34 = 1;
            v21 = *(_DWORD *)(a1 + 24) + 1;
            v10 = v37;
            v13 = (__int64 *)(v31 + 24LL * v33);
            v35 = *v13;
            if ( a2 != *v13 )
            {
              while ( v35 != -4096 )
              {
                if ( !v32 && v35 == -8192 )
                  v32 = v13;
                v33 = v30 & (v34 + v33);
                v13 = (__int64 *)(v31 + 24LL * v33);
                v35 = *v13;
                if ( a2 == *v13 )
                  goto LABEL_18;
                ++v34;
              }
              if ( v32 )
                v13 = v32;
            }
            goto LABEL_18;
          }
LABEL_45:
          ++*(_DWORD *)(a1 + 24);
          BUG();
        }
      }
      else
      {
        ++*(_QWORD *)(a1 + 8);
      }
      v36 = v10;
      sub_337DA20(a1 + 8, 2 * v7);
      v22 = *(_DWORD *)(a1 + 32);
      if ( v22 )
      {
        v23 = v22 - 1;
        v24 = *(_QWORD *)(a1 + 16);
        v25 = (v22 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
        v21 = *(_DWORD *)(a1 + 24) + 1;
        v10 = v36;
        v13 = (__int64 *)(v24 + 24LL * v25);
        v26 = *v13;
        if ( a2 != *v13 )
        {
          v27 = 1;
          v28 = 0;
          while ( v26 != -4096 )
          {
            if ( !v28 && v26 == -8192 )
              v28 = v13;
            v25 = v23 & (v27 + v25);
            v13 = (__int64 *)(v24 + 24LL * v25);
            v26 = *v13;
            if ( a2 == *v13 )
              goto LABEL_18;
            ++v27;
          }
          if ( v28 )
            v13 = v28;
        }
        goto LABEL_18;
      }
      goto LABEL_45;
    }
    v38[0] = a2;
    v17 = sub_338B750(a1, (__int64)v4);
    v19 = v18;
    result = sub_337DC20(a1 + 8, v38);
    *result = v17;
    *((_DWORD *)result + 2) = v19;
  }
  return result;
}
