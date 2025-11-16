// Function: sub_1A930A0
// Address: 0x1a930a0
//
__int64 __fastcall sub_1A930A0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v6; // rbx
  __int64 result; // rax
  _BYTE *v9; // rsi
  __int64 v10; // r12
  unsigned __int64 v11; // r14
  unsigned int i; // r12d
  __int64 v13; // r8
  int v14; // r11d
  __int64 *v15; // r10
  unsigned int v16; // edx
  __int64 *v17; // rdi
  __int64 v18; // rcx
  unsigned int v19; // esi
  int v20; // ecx
  int v21; // ecx
  __int64 v22; // rdi
  unsigned int v23; // r9d
  __int64 v24; // rsi
  int v25; // edx
  int v26; // r8d
  __int64 *v27; // r11
  int v28; // edi
  _BYTE *v29; // rsi
  int v30; // ecx
  int v31; // ecx
  __int64 v32; // rdi
  __int64 *v33; // r11
  int v34; // r8d
  unsigned int v35; // r9d
  unsigned int v37; // [rsp+20h] [rbp-60h]
  __int64 v38; // [rsp+30h] [rbp-50h]
  __int64 v39; // [rsp+38h] [rbp-48h]
  _QWORD v40[7]; // [rsp+48h] [rbp-38h] BYREF

  v6 = a1 + 24;
  result = a2 + 24;
  if ( !a2 )
    result = 0;
  v39 = *(_QWORD *)(a1 + 40) + 40LL;
  v38 = result;
  if ( v6 != result )
  {
    result = (__int64)v40;
    if ( v6 != *(_QWORD *)(a1 + 40) + 40LL )
    {
      while ( 1 )
      {
        if ( *(_BYTE *)(v6 - 8) == 78 )
        {
          v40[0] = v6 - 24;
          v9 = *(_BYTE **)(a3 + 8);
          if ( v9 == *(_BYTE **)(a3 + 16) )
          {
            sub_187FFB0(a3, v9, v40);
          }
          else
          {
            if ( v9 )
            {
              *(_QWORD *)v9 = v6 - 24;
              v9 = *(_BYTE **)(a3 + 8);
            }
            *(_QWORD *)(a3 + 8) = v9 + 8;
          }
        }
        result = (unsigned int)*(unsigned __int8 *)(v6 - 8) - 25;
        if ( (unsigned int)result > 9
          || (v10 = *(_QWORD *)(v6 + 16), (result = sub_157EBA0(v10)) == 0)
          || (v37 = sub_15F4D60(result), v11 = sub_157EBA0(v10), (result = v37) == 0) )
        {
          v6 = *(_QWORD *)(v6 + 8);
          if ( v6 == v39 )
            return result;
          goto LABEL_12;
        }
        for ( i = 0; i != v37; ++i )
        {
          while ( 1 )
          {
            result = sub_15F4DF0(v11, i);
            v19 = *(_DWORD *)(a4 + 24);
            v40[0] = result;
            if ( !v19 )
            {
              ++*(_QWORD *)a4;
LABEL_22:
              sub_13B3D40(a4, 2 * v19);
              v20 = *(_DWORD *)(a4 + 24);
              if ( !v20 )
                goto LABEL_65;
              result = v40[0];
              v21 = v20 - 1;
              v22 = *(_QWORD *)(a4 + 8);
              v23 = v21 & ((LODWORD(v40[0]) >> 9) ^ (LODWORD(v40[0]) >> 4));
              v15 = (__int64 *)(v22 + 8LL * v23);
              v24 = *v15;
              v25 = *(_DWORD *)(a4 + 16) + 1;
              if ( *v15 != v40[0] )
              {
                v26 = 1;
                v27 = 0;
                while ( v24 != -8 )
                {
                  if ( !v27 && v24 == -16 )
                    v27 = v15;
                  v23 = v21 & (v26 + v23);
                  v15 = (__int64 *)(v22 + 8LL * v23);
                  v24 = *v15;
                  if ( v40[0] == *v15 )
                    goto LABEL_38;
                  ++v26;
                }
                if ( v27 )
                  v15 = v27;
              }
              goto LABEL_38;
            }
            v13 = *(_QWORD *)(a4 + 8);
            v14 = 1;
            v15 = 0;
            v16 = (v19 - 1) & (((unsigned int)result >> 9) ^ ((unsigned int)result >> 4));
            v17 = (__int64 *)(v13 + 8LL * v16);
            v18 = *v17;
            if ( result != *v17 )
              break;
LABEL_19:
            if ( v37 == ++i )
              goto LABEL_44;
          }
          while ( v18 != -8 )
          {
            if ( v15 || v18 != -16 )
              v17 = v15;
            v16 = (v19 - 1) & (v14 + v16);
            v18 = *(_QWORD *)(v13 + 8LL * v16);
            if ( result == v18 )
              goto LABEL_19;
            ++v14;
            v15 = v17;
            v17 = (__int64 *)(v13 + 8LL * v16);
          }
          if ( !v15 )
            v15 = v17;
          v28 = *(_DWORD *)(a4 + 16);
          ++*(_QWORD *)a4;
          v25 = v28 + 1;
          if ( 4 * (v28 + 1) >= 3 * v19 )
            goto LABEL_22;
          if ( v19 - *(_DWORD *)(a4 + 20) - v25 <= v19 >> 3 )
          {
            sub_13B3D40(a4, v19);
            v30 = *(_DWORD *)(a4 + 24);
            if ( !v30 )
            {
LABEL_65:
              ++*(_DWORD *)(a4 + 16);
              BUG();
            }
            v31 = v30 - 1;
            v32 = *(_QWORD *)(a4 + 8);
            v33 = 0;
            v34 = 1;
            v25 = *(_DWORD *)(a4 + 16) + 1;
            v35 = v31 & ((LODWORD(v40[0]) >> 9) ^ (LODWORD(v40[0]) >> 4));
            v15 = (__int64 *)(v32 + 8LL * v35);
            result = *v15;
            if ( v40[0] != *v15 )
            {
              while ( result != -8 )
              {
                if ( !v33 && result == -16 )
                  v33 = v15;
                v35 = v31 & (v34 + v35);
                v15 = (__int64 *)(v32 + 8LL * v35);
                result = *v15;
                if ( v40[0] == *v15 )
                  goto LABEL_38;
                ++v34;
              }
              result = v40[0];
              if ( v33 )
                v15 = v33;
            }
          }
LABEL_38:
          *(_DWORD *)(a4 + 16) = v25;
          if ( *v15 != -8 )
            --*(_DWORD *)(a4 + 20);
          *v15 = result;
          v29 = *(_BYTE **)(a5 + 8);
          if ( v29 == *(_BYTE **)(a5 + 16) )
          {
            result = (__int64)sub_1292090(a5, v29, v40);
            goto LABEL_19;
          }
          if ( v29 )
          {
            result = v40[0];
            *(_QWORD *)v29 = v40[0];
            v29 = *(_BYTE **)(a5 + 8);
          }
          *(_QWORD *)(a5 + 8) = v29 + 8;
        }
LABEL_44:
        v6 = *(_QWORD *)(v6 + 8);
        if ( v6 == v39 )
          return result;
LABEL_12:
        if ( v38 == v6 )
          return result;
        if ( !v6 )
          BUG();
      }
    }
  }
  return result;
}
