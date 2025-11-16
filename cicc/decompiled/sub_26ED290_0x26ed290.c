// Function: sub_26ED290
// Address: 0x26ed290
//
unsigned __int64 __fastcall sub_26ED290(__int64 *a1, __int64 a2)
{
  _QWORD *v2; // rsi
  unsigned __int64 result; // rax
  __int64 v4; // r15
  int v5; // r14d
  unsigned int v6; // r12d
  __int64 v7; // r8
  unsigned int v8; // edi
  __int64 *v9; // rcx
  __int64 v10; // rdx
  __int64 v11; // rbx
  __int64 v12; // rax
  __int64 v13; // rcx
  __int64 v14; // r13
  unsigned int v15; // esi
  int v16; // ecx
  int v17; // ecx
  __int64 v18; // rdi
  __int64 *v19; // r10
  __int64 v20; // rsi
  int v21; // edx
  int v22; // r11d
  int v23; // edx
  int v24; // ecx
  int v25; // ecx
  __int64 v26; // rdi
  int v27; // r11d
  __int64 *v28; // r8
  __int64 v29; // rsi
  int v30; // r11d

  v2 = (_QWORD *)(a2 + 48);
  result = *v2 & 0xFFFFFFFFFFFFFFF8LL;
  if ( (_QWORD *)result != v2 )
  {
    if ( !result )
      BUG();
    v4 = result - 24;
    result = (unsigned int)*(unsigned __int8 *)(result - 24) - 30;
    if ( (unsigned int)result <= 0xA )
    {
      result = sub_B46E30(v4);
      v5 = result;
      if ( (_DWORD)result )
      {
        v6 = 0;
        while ( 1 )
        {
          v11 = sub_B46EC0(v4, v6);
          v12 = sub_AA4FF0(v11);
          if ( !v12 )
            BUG();
          result = (unsigned int)*(unsigned __int8 *)(v12 - 24) - 39;
          if ( (unsigned int)result <= 0x38 )
          {
            v13 = 0x100060000000001LL;
            if ( _bittest64(&v13, result) )
              goto LABEL_7;
          }
          v14 = *a1;
          v15 = *(_DWORD *)(*a1 + 24);
          if ( !v15 )
          {
            ++*(_QWORD *)v14;
            goto LABEL_13;
          }
          v7 = *(_QWORD *)(v14 + 8);
          result = ((unsigned int)v11 >> 4) ^ ((unsigned int)v11 >> 9);
          v8 = (v15 - 1) & (((unsigned int)v11 >> 4) ^ ((unsigned int)v11 >> 9));
          v9 = (__int64 *)(v7 + 8LL * v8);
          v10 = *v9;
          if ( v11 == *v9 )
          {
LABEL_7:
            if ( v5 == ++v6 )
              return result;
          }
          else
          {
            v22 = 1;
            v19 = 0;
            while ( v10 != -4096 )
            {
              if ( v10 != -8192 || v19 )
                v9 = v19;
              v8 = (v15 - 1) & (v22 + v8);
              v10 = *(_QWORD *)(v7 + 8LL * v8);
              if ( v11 == v10 )
                goto LABEL_7;
              ++v22;
              v19 = v9;
              v9 = (__int64 *)(v7 + 8LL * v8);
            }
            v23 = *(_DWORD *)(v14 + 16);
            if ( !v19 )
              v19 = v9;
            ++*(_QWORD *)v14;
            v21 = v23 + 1;
            if ( 4 * v21 < 3 * v15 )
            {
              if ( v15 - *(_DWORD *)(v14 + 20) - v21 > v15 >> 3 )
                goto LABEL_15;
              sub_CF28B0(v14, v15);
              v24 = *(_DWORD *)(v14 + 24);
              if ( !v24 )
              {
LABEL_49:
                ++*(_DWORD *)(v14 + 16);
                BUG();
              }
              v25 = v24 - 1;
              v26 = *(_QWORD *)(v14 + 8);
              v27 = 1;
              v28 = 0;
              result = v25 & (((unsigned int)v11 >> 4) ^ ((unsigned int)v11 >> 9));
              v19 = (__int64 *)(v26 + 8 * result);
              v29 = *v19;
              v21 = *(_DWORD *)(v14 + 16) + 1;
              if ( v11 == *v19 )
                goto LABEL_15;
              while ( v29 != -4096 )
              {
                if ( v29 == -8192 && !v28 )
                  v28 = v19;
                result = v25 & (unsigned int)(result + v27);
                v19 = (__int64 *)(v26 + 8 * result);
                v29 = *v19;
                if ( v11 == *v19 )
                  goto LABEL_15;
                ++v27;
              }
              goto LABEL_28;
            }
LABEL_13:
            sub_CF28B0(v14, 2 * v15);
            v16 = *(_DWORD *)(v14 + 24);
            if ( !v16 )
              goto LABEL_49;
            v17 = v16 - 1;
            v18 = *(_QWORD *)(v14 + 8);
            result = v17 & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
            v19 = (__int64 *)(v18 + 8 * result);
            v20 = *v19;
            v21 = *(_DWORD *)(v14 + 16) + 1;
            if ( *v19 == v11 )
              goto LABEL_15;
            v30 = 1;
            v28 = 0;
            while ( v20 != -4096 )
            {
              if ( v20 == -8192 && !v28 )
                v28 = v19;
              result = v17 & (unsigned int)(result + v30);
              v19 = (__int64 *)(v18 + 8 * result);
              v20 = *v19;
              if ( v11 == *v19 )
                goto LABEL_15;
              ++v30;
            }
LABEL_28:
            if ( v28 )
              v19 = v28;
LABEL_15:
            *(_DWORD *)(v14 + 16) = v21;
            if ( *v19 != -4096 )
              --*(_DWORD *)(v14 + 20);
            ++v6;
            *v19 = v11;
            if ( v5 == v6 )
              return result;
          }
        }
      }
    }
  }
  return result;
}
