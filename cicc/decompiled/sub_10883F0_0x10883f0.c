// Function: sub_10883F0
// Address: 0x10883f0
//
__int64 *__fastcall sub_10883F0(__int64 a1, __int64 a2)
{
  _DWORD *v3; // r12
  int v4; // eax
  __int64 *v5; // r12
  unsigned int v6; // ecx
  __int64 *result; // rax
  unsigned int v8; // r14d
  int v9; // ecx
  __int64 v10; // r9
  __int64 *v11; // rdx
  unsigned int v12; // r8d
  __int64 v13; // rdi
  __int64 v14; // r15
  int v15; // eax
  int v16; // edx
  __int64 v17; // rsi
  __int64 v18; // rax
  unsigned int v19; // esi
  __int64 v20; // r13
  int v21; // edi
  int v22; // edi
  __int64 v23; // r9
  __int64 v24; // rsi
  __int64 v25; // r8
  int v26; // eax
  int v27; // esi
  int v28; // esi
  __int64 v29; // r8
  __int64 *v30; // r9
  __int64 v31; // r15
  int v32; // r11d
  __int64 v33; // rdi
  __int64 v34; // rdx
  int v35; // r15d
  __int64 *v36; // r10
  __int64 v37; // [rsp+8h] [rbp-48h]
  __int64 *v39; // [rsp+18h] [rbp-38h]

  v3 = *(_DWORD **)(a1 + 8);
  v4 = (*(__int64 (__fastcall **)(_DWORD *))(*(_QWORD *)v3 + 80LL))(v3) + v3[8] - v3[4];
  v5 = *(__int64 **)(a2 + 40);
  v6 = (*(_BYTE *)(a1 + 240) == 0 ? 0xFFFFFFDC : 0) + v4 + 40 * *(_DWORD *)(a1 + 28) + 56;
  result = &v5[*(unsigned int *)(a2 + 48)];
  v39 = result;
  if ( v5 != result )
  {
    v8 = v6;
    v37 = a1 + 144;
    while ( 1 )
    {
      v19 = *(_DWORD *)(a1 + 168);
      v20 = *v5;
      if ( !v19 )
        break;
      v9 = 1;
      v10 = *(_QWORD *)(a1 + 152);
      v11 = 0;
      v12 = (v19 - 1) & (((unsigned int)v20 >> 9) ^ ((unsigned int)v20 >> 4));
      result = (__int64 *)(v10 + 16LL * v12);
      v13 = *result;
      if ( v20 == *result )
      {
LABEL_4:
        v14 = result[1];
        if ( v14 && *(_DWORD *)(v14 + 72) != -1 )
        {
          v15 = sub_E5CAC0((__int64 *)a2, *v5);
          *(_DWORD *)(v14 + 16) = v15;
          v16 = v15;
          if ( *(char *)(v14 + 36) >= 0 )
          {
            *(_DWORD *)(v14 + 20) = v8;
            v8 += v15;
          }
          v17 = *(_QWORD *)(v14 + 104);
          v18 = *(_QWORD *)(v14 + 96);
          if ( v17 != v18 )
          {
            if ( (unsigned __int64)(v17 - v18) <= 0x17FFD0 )
            {
              *(_WORD *)(v14 + 32) = -21845 * ((v17 - v18) >> 3);
              *(_DWORD *)(v14 + 24) = v8;
            }
            else
            {
              *(_DWORD *)(v14 + 24) = v8;
              v8 += 10;
              *(_WORD *)(v14 + 32) = -1;
            }
            v8 -= 1431655762 * ((v17 - v18) >> 3);
            do
            {
              while ( *(_WORD *)(a1 + 24) != 358 || *(_WORD *)(v18 + 8) != 37 )
              {
                v34 = *(_QWORD *)(v18 + 16);
                v18 += 24;
                *(_DWORD *)(v18 - 20) = *(_DWORD *)(v34 + 56);
                if ( v17 == v18 )
                  goto LABEL_43;
              }
              v18 += 24;
            }
            while ( v17 != v18 );
LABEL_43:
            v16 = *(_DWORD *)(v14 + 16);
          }
          result = *(__int64 **)(*(_QWORD *)(v14 + 88) + 64LL);
          *((_DWORD *)result + 1) = v16;
          *((_WORD *)result + 4) = *(_WORD *)(v14 + 32);
          *((_WORD *)result + 5) = *(_WORD *)(v14 + 34);
        }
        if ( v39 == ++v5 )
        {
LABEL_18:
          v6 = v8;
          goto LABEL_19;
        }
      }
      else
      {
        while ( v13 != -4096 )
        {
          if ( !v11 && v13 == -8192 )
            v11 = result;
          v12 = (v19 - 1) & (v9 + v12);
          result = (__int64 *)(v10 + 16LL * v12);
          v13 = *result;
          if ( v20 == *result )
            goto LABEL_4;
          ++v9;
        }
        if ( !v11 )
          v11 = result;
        v26 = *(_DWORD *)(a1 + 160);
        ++*(_QWORD *)(a1 + 144);
        result = (__int64 *)(unsigned int)(v26 + 1);
        if ( 4 * (int)result < 3 * v19 )
        {
          if ( v19 - *(_DWORD *)(a1 + 164) - (unsigned int)result <= v19 >> 3 )
          {
            sub_10854B0(v37, v19);
            v27 = *(_DWORD *)(a1 + 168);
            if ( !v27 )
            {
LABEL_60:
              ++*(_DWORD *)(a1 + 160);
              BUG();
            }
            v28 = v27 - 1;
            v29 = *(_QWORD *)(a1 + 152);
            v30 = 0;
            LODWORD(v31) = v28 & (((unsigned int)v20 >> 9) ^ ((unsigned int)v20 >> 4));
            v32 = 1;
            result = (__int64 *)(unsigned int)(*(_DWORD *)(a1 + 160) + 1);
            v11 = (__int64 *)(v29 + 16LL * (unsigned int)v31);
            v33 = *v11;
            if ( v20 != *v11 )
            {
              while ( v33 != -4096 )
              {
                if ( !v30 && v33 == -8192 )
                  v30 = v11;
                v31 = v28 & (unsigned int)(v31 + v32);
                v11 = (__int64 *)(v29 + 16 * v31);
                v33 = *v11;
                if ( v20 == *v11 )
                  goto LABEL_15;
                ++v32;
              }
              if ( v30 )
                v11 = v30;
            }
          }
          goto LABEL_15;
        }
LABEL_13:
        sub_10854B0(v37, 2 * v19);
        v21 = *(_DWORD *)(a1 + 168);
        if ( !v21 )
          goto LABEL_60;
        v22 = v21 - 1;
        v23 = *(_QWORD *)(a1 + 152);
        LODWORD(v24) = v22 & (((unsigned int)v20 >> 9) ^ ((unsigned int)v20 >> 4));
        result = (__int64 *)(unsigned int)(*(_DWORD *)(a1 + 160) + 1);
        v11 = (__int64 *)(v23 + 16LL * (unsigned int)v24);
        v25 = *v11;
        if ( v20 != *v11 )
        {
          v35 = 1;
          v36 = 0;
          while ( v25 != -4096 )
          {
            if ( v25 == -8192 && !v36 )
              v36 = v11;
            v24 = v22 & (unsigned int)(v24 + v35);
            v11 = (__int64 *)(v23 + 16 * v24);
            v25 = *v11;
            if ( v20 == *v11 )
              goto LABEL_15;
            ++v35;
          }
          if ( v36 )
            v11 = v36;
        }
LABEL_15:
        *(_DWORD *)(a1 + 160) = (_DWORD)result;
        if ( *v11 != -4096 )
          --*(_DWORD *)(a1 + 164);
        *v11 = v20;
        ++v5;
        v11[1] = 0;
        if ( v39 == v5 )
          goto LABEL_18;
      }
    }
    ++*(_QWORD *)(a1 + 144);
    goto LABEL_13;
  }
LABEL_19:
  *(_DWORD *)(a1 + 36) = v6;
  return result;
}
