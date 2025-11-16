// Function: sub_1247D30
// Address: 0x1247d30
//
__int64 __fastcall sub_1247D30(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int *a5)
{
  __int64 v8; // rbx
  __int64 result; // rax
  __int64 v10; // r14
  unsigned int v11; // esi
  int *v12; // r15
  int v13; // r13d
  __int64 v14; // r9
  unsigned int v15; // edi
  int v16; // ecx
  __int64 v17; // rdx
  __int64 v18; // rcx
  int v19; // r11d
  __int64 v20; // rdx
  int v21; // eax
  int v22; // ecx
  int v23; // eax
  int v24; // esi
  __int64 v25; // r8
  int v26; // edi
  int v27; // r10d
  __int64 v28; // r9
  int v29; // eax
  __int64 v30; // rdi
  int v31; // r10d
  unsigned int v32; // r8d
  int v33; // esi
  int v34; // [rsp+0h] [rbp-40h]
  __int64 v35; // [rsp+8h] [rbp-38h]

  *(_QWORD *)(a1 + 40) = a1 + 24;
  *(_QWORD *)(a1 + 48) = a1 + 24;
  *(_QWORD *)(a1 + 88) = a1 + 72;
  *(_QWORD *)(a1 + 96) = a1 + 72;
  *(_QWORD *)a1 = a2;
  *(_QWORD *)(a1 + 8) = a3;
  *(_DWORD *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 32) = 0;
  *(_QWORD *)(a1 + 56) = 0;
  *(_DWORD *)(a1 + 72) = 0;
  *(_QWORD *)(a1 + 80) = 0;
  v35 = a1 + 112;
  *(_QWORD *)(a1 + 104) = 0;
  *(_QWORD *)(a1 + 112) = 0;
  *(_QWORD *)(a1 + 120) = 0;
  *(_QWORD *)(a1 + 128) = 0;
  *(_DWORD *)(a1 + 136) = 0;
  *(_DWORD *)(a1 + 144) = 0;
  *(_DWORD *)(a1 + 152) = a4;
  if ( (*(_BYTE *)(a3 + 2) & 1) != 0 )
  {
    sub_B2C6D0(a3, a2, a3, a4);
    v8 = *(_QWORD *)(a3 + 96);
    result = 5LL * *(_QWORD *)(a3 + 104);
    v10 = v8 + 40LL * *(_QWORD *)(a3 + 104);
    if ( (*(_BYTE *)(a3 + 2) & 1) != 0 )
    {
      result = sub_B2C6D0(a3, a2, v17, v18);
      v8 = *(_QWORD *)(a3 + 96);
    }
  }
  else
  {
    v8 = *(_QWORD *)(a3 + 96);
    result = 5LL * *(_QWORD *)(a3 + 104);
    v10 = v8 + 40LL * *(_QWORD *)(a3 + 104);
  }
  if ( v8 != v10 )
  {
    while ( 1 )
    {
      while ( (*(_BYTE *)(v8 + 7) & 0x10) != 0 )
      {
        v8 += 40;
        if ( v10 == v8 )
          return result;
      }
      v11 = *(_DWORD *)(a1 + 136);
      v12 = a5 + 1;
      v13 = *a5;
      if ( !v11 )
        break;
      v14 = *(_QWORD *)(a1 + 120);
      v15 = (v11 - 1) & (37 * v13);
      result = v14 + 16LL * v15;
      v16 = *(_DWORD *)result;
      if ( v13 != *(_DWORD *)result )
      {
        v19 = 1;
        v20 = 0;
        while ( v16 != -1 )
        {
          if ( v20 || v16 != -2 )
            result = v20;
          v15 = (v11 - 1) & (v19 + v15);
          v16 = *(_DWORD *)(v14 + 16LL * v15);
          if ( v13 == v16 )
            goto LABEL_9;
          ++v19;
          v20 = result;
          result = v14 + 16LL * v15;
        }
        if ( !v20 )
          v20 = result;
        v21 = *(_DWORD *)(a1 + 128);
        ++*(_QWORD *)(a1 + 112);
        v22 = v21 + 1;
        if ( 4 * (v21 + 1) < 3 * v11 )
        {
          result = v11 - *(_DWORD *)(a1 + 132) - v22;
          if ( (unsigned int)result <= v11 >> 3 )
          {
            v34 = 37 * v13;
            sub_1247200(v35, v11);
            v29 = *(_DWORD *)(a1 + 136);
            if ( !v29 )
            {
LABEL_48:
              ++*(_DWORD *)(a1 + 128);
              BUG();
            }
            result = (unsigned int)(v29 - 1);
            v30 = *(_QWORD *)(a1 + 120);
            v28 = 0;
            v31 = 1;
            v32 = result & v34;
            v22 = *(_DWORD *)(a1 + 128) + 1;
            v20 = v30 + 16LL * ((unsigned int)result & v34);
            v33 = *(_DWORD *)v20;
            if ( v13 != *(_DWORD *)v20 )
            {
              while ( v33 != -1 )
              {
                if ( !v28 && v33 == -2 )
                  v28 = v20;
                v32 = result & (v31 + v32);
                v20 = v30 + 16LL * v32;
                v33 = *(_DWORD *)v20;
                if ( v13 == *(_DWORD *)v20 )
                  goto LABEL_19;
                ++v31;
              }
              goto LABEL_35;
            }
          }
          goto LABEL_19;
        }
LABEL_23:
        sub_1247200(v35, 2 * v11);
        v23 = *(_DWORD *)(a1 + 136);
        if ( !v23 )
          goto LABEL_48;
        v24 = v23 - 1;
        v25 = *(_QWORD *)(a1 + 120);
        result = (v23 - 1) & (unsigned int)(37 * v13);
        v22 = *(_DWORD *)(a1 + 128) + 1;
        v20 = v25 + 16 * result;
        v26 = *(_DWORD *)v20;
        if ( v13 != *(_DWORD *)v20 )
        {
          v27 = 1;
          v28 = 0;
          while ( v26 != -1 )
          {
            if ( v26 == -2 && !v28 )
              v28 = v20;
            result = v24 & (unsigned int)(v27 + result);
            v20 = v25 + 16LL * (unsigned int)result;
            v26 = *(_DWORD *)v20;
            if ( v13 == *(_DWORD *)v20 )
              goto LABEL_19;
            ++v27;
          }
LABEL_35:
          if ( v28 )
            v20 = v28;
        }
LABEL_19:
        *(_DWORD *)(a1 + 128) = v22;
        if ( *(_DWORD *)v20 != -1 )
          --*(_DWORD *)(a1 + 132);
        *(_DWORD *)v20 = v13;
        *(_QWORD *)(v20 + 8) = v8;
      }
LABEL_9:
      v8 += 40;
      *(_DWORD *)(a1 + 144) = v13 + 1;
      a5 = v12;
      if ( v10 == v8 )
        return result;
    }
    ++*(_QWORD *)(a1 + 112);
    goto LABEL_23;
  }
  return result;
}
