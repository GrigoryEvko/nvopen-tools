// Function: sub_30BB990
// Address: 0x30bb990
//
__int64 __fastcall sub_30BB990(__int64 a1)
{
  __int64 *v1; // rdx
  __int64 result; // rax
  __int64 v4; // r14
  __int64 v5; // r12
  __int64 v6; // r10
  __int64 v7; // r8
  unsigned int v8; // edi
  _QWORD *v9; // rax
  __int64 v10; // rcx
  unsigned int v11; // esi
  __int64 v12; // rbx
  __int64 v13; // r11
  int v14; // esi
  int v15; // esi
  __int64 v16; // r8
  unsigned int v17; // ecx
  int v18; // eax
  _QWORD *v19; // rdx
  __int64 v20; // rdi
  int v21; // eax
  int v22; // ecx
  int v23; // ecx
  __int64 v24; // rdi
  _QWORD *v25; // r8
  unsigned int v26; // r15d
  int v27; // r9d
  __int64 v28; // rsi
  int v29; // r15d
  _QWORD *v30; // r9
  __int64 v31; // [rsp+8h] [rbp-58h]
  __int64 v32; // [rsp+8h] [rbp-58h]
  __int64 v33; // [rsp+10h] [rbp-50h]
  int v34; // [rsp+10h] [rbp-50h]
  __int64 v35; // [rsp+10h] [rbp-50h]
  __int64 v36; // [rsp+18h] [rbp-48h]
  __int64 v37; // [rsp+20h] [rbp-40h]
  __int64 v38; // [rsp+28h] [rbp-38h]

  v1 = *(__int64 **)(a1 + 24);
  v36 = a1 + 64;
  result = *v1;
  v38 = *v1;
  v37 = *v1 + 8LL * *((unsigned int *)v1 + 2);
  if ( v37 != *v1 )
  {
    v4 = 1;
    while ( 1 )
    {
      v5 = *(_QWORD *)(*(_QWORD *)v38 + 56LL);
      v6 = *(_QWORD *)v38 + 48LL;
      if ( v6 != v5 )
        break;
LABEL_16:
      v38 += 8;
      result = v38;
      if ( v37 == v38 )
        return result;
    }
    while ( 1 )
    {
      v11 = *(_DWORD *)(a1 + 88);
      v12 = v5 - 24;
      v13 = v4;
      if ( !v5 )
        v12 = 0;
      ++v4;
      if ( !v11 )
        break;
      v7 = *(_QWORD *)(a1 + 72);
      v8 = (v11 - 1) & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
      v9 = (_QWORD *)(v7 + 16LL * v8);
      v10 = *v9;
      if ( v12 == *v9 )
      {
LABEL_6:
        v5 = *(_QWORD *)(v5 + 8);
        if ( v6 == v5 )
          goto LABEL_16;
      }
      else
      {
        v34 = 1;
        v19 = 0;
        while ( v10 != -4096 )
        {
          if ( v10 != -8192 || v19 )
            v9 = v19;
          v8 = (v11 - 1) & (v34 + v8);
          v10 = *(_QWORD *)(v7 + 16LL * v8);
          if ( v12 == v10 )
            goto LABEL_6;
          ++v34;
          v19 = v9;
          v9 = (_QWORD *)(v7 + 16LL * v8);
        }
        if ( !v19 )
          v19 = v9;
        v21 = *(_DWORD *)(a1 + 80);
        ++*(_QWORD *)(a1 + 64);
        v18 = v21 + 1;
        if ( 4 * v18 < 3 * v11 )
        {
          if ( v11 - *(_DWORD *)(a1 + 84) - v18 <= v11 >> 3 )
          {
            v32 = v13;
            v35 = v6;
            sub_30BB7B0(v36, v11);
            v22 = *(_DWORD *)(a1 + 88);
            if ( !v22 )
            {
LABEL_50:
              ++*(_DWORD *)(a1 + 80);
              BUG();
            }
            v23 = v22 - 1;
            v24 = *(_QWORD *)(a1 + 72);
            v25 = 0;
            v26 = v23 & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
            v6 = v35;
            v13 = v32;
            v27 = 1;
            v18 = *(_DWORD *)(a1 + 80) + 1;
            v19 = (_QWORD *)(v24 + 16LL * v26);
            v28 = *v19;
            if ( v12 != *v19 )
            {
              while ( v28 != -4096 )
              {
                if ( !v25 && v28 == -8192 )
                  v25 = v19;
                v26 = v23 & (v27 + v26);
                v19 = (_QWORD *)(v24 + 16LL * v26);
                v28 = *v19;
                if ( v12 == *v19 )
                  goto LABEL_13;
                ++v27;
              }
              if ( v25 )
                v19 = v25;
            }
          }
          goto LABEL_13;
        }
LABEL_11:
        v31 = v13;
        v33 = v6;
        sub_30BB7B0(v36, 2 * v11);
        v14 = *(_DWORD *)(a1 + 88);
        if ( !v14 )
          goto LABEL_50;
        v15 = v14 - 1;
        v16 = *(_QWORD *)(a1 + 72);
        v6 = v33;
        v13 = v31;
        v17 = v15 & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
        v18 = *(_DWORD *)(a1 + 80) + 1;
        v19 = (_QWORD *)(v16 + 16LL * v17);
        v20 = *v19;
        if ( v12 != *v19 )
        {
          v29 = 1;
          v30 = 0;
          while ( v20 != -4096 )
          {
            if ( v20 == -8192 && !v30 )
              v30 = v19;
            v17 = v15 & (v29 + v17);
            v19 = (_QWORD *)(v16 + 16LL * v17);
            v20 = *v19;
            if ( v12 == *v19 )
              goto LABEL_13;
            ++v29;
          }
          if ( v30 )
            v19 = v30;
        }
LABEL_13:
        *(_DWORD *)(a1 + 80) = v18;
        if ( *v19 != -4096 )
          --*(_DWORD *)(a1 + 84);
        *v19 = v12;
        v19[1] = v13;
        v5 = *(_QWORD *)(v5 + 8);
        if ( v6 == v5 )
          goto LABEL_16;
      }
    }
    ++*(_QWORD *)(a1 + 64);
    goto LABEL_11;
  }
  return result;
}
