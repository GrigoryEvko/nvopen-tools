// Function: sub_28E74C0
// Address: 0x28e74c0
//
unsigned __int64 __fastcall sub_28E74C0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // rbx
  __int64 v6; // rdi
  unsigned __int64 result; // rax
  _BYTE *v10; // rsi
  __int64 v11; // rdx
  __int64 v12; // r13
  unsigned int v13; // r12d
  int v14; // r14d
  __int64 v15; // r8
  int v16; // r11d
  unsigned __int64 *v17; // r10
  unsigned int v18; // edx
  unsigned __int64 *v19; // rdi
  unsigned __int64 v20; // rcx
  unsigned int v21; // esi
  int v22; // ecx
  unsigned __int64 v23; // rsi
  int v24; // ecx
  __int64 v25; // rdi
  int v26; // edx
  unsigned int v27; // r9d
  int v28; // r8d
  unsigned __int64 *v29; // r11
  int v30; // edi
  _BYTE *v31; // rsi
  int v32; // ecx
  int v33; // ecx
  __int64 v34; // rdi
  int v35; // r8d
  unsigned int v36; // r9d
  __int64 v38; // [rsp+18h] [rbp-68h]
  __int64 v39; // [rsp+30h] [rbp-50h]
  unsigned __int64 v40; // [rsp+38h] [rbp-48h]
  _QWORD v41[7]; // [rsp+48h] [rbp-38h] BYREF

  v5 = a1 + 24;
  v6 = *(_QWORD *)(a1 + 40) + 48LL;
  result = a2 + 24;
  if ( !a2 )
    result = 0;
  v39 = v6;
  v40 = result;
  if ( v5 != v6 && v5 != result )
  {
LABEL_5:
    if ( *(_BYTE *)(v5 - 24) == 85 )
    {
      v41[0] = v5 - 24;
      v10 = *(_BYTE **)(a3 + 8);
      if ( v10 == *(_BYTE **)(a3 + 16) )
      {
        sub_2628C60(a3, v10, v41);
      }
      else
      {
        if ( v10 )
        {
          *(_QWORD *)v10 = v5 - 24;
          v10 = *(_BYTE **)(a3 + 8);
        }
        *(_QWORD *)(a3 + 8) = v10 + 8;
      }
    }
    result = (unsigned int)*(unsigned __int8 *)(v5 - 24) - 30;
    if ( (unsigned int)result > 0xA )
      goto LABEL_11;
    v11 = *(_QWORD *)(v5 + 16);
    result = *(_QWORD *)(v11 + 48) & 0xFFFFFFFFFFFFFFF8LL;
    if ( result == v11 + 48 )
      goto LABEL_11;
    if ( !result )
      goto LABEL_14;
    v12 = result - 24;
    result = (unsigned int)*(unsigned __int8 *)(result - 24) - 30;
    if ( (unsigned int)result > 0xA || (result = sub_B46E30(v12), !(_DWORD)result) )
    {
LABEL_11:
      v5 = *(_QWORD *)(v5 + 8);
      if ( v40 == v5 )
        return result;
      goto LABEL_12;
    }
    v13 = 0;
    v38 = a3;
    v14 = result;
    while ( 1 )
    {
      result = sub_B46EC0(v12, v13);
      v21 = *(_DWORD *)(a4 + 24);
      v41[0] = result;
      if ( !v21 )
        break;
      v15 = *(_QWORD *)(a4 + 8);
      v16 = 1;
      v17 = 0;
      v18 = (v21 - 1) & (((unsigned int)result >> 9) ^ ((unsigned int)result >> 4));
      v19 = (unsigned __int64 *)(v15 + 8LL * v18);
      v20 = *v19;
      if ( result == *v19 )
      {
LABEL_21:
        if ( v14 == ++v13 )
          goto LABEL_46;
      }
      else
      {
        while ( v20 != -4096 )
        {
          if ( v17 || v20 != -8192 )
            v19 = v17;
          v18 = (v21 - 1) & (v16 + v18);
          v20 = *(_QWORD *)(v15 + 8LL * v18);
          if ( result == v20 )
            goto LABEL_21;
          ++v16;
          v17 = v19;
          v19 = (unsigned __int64 *)(v15 + 8LL * v18);
        }
        if ( !v17 )
          v17 = v19;
        v30 = *(_DWORD *)(a4 + 16);
        ++*(_QWORD *)a4;
        v26 = v30 + 1;
        if ( 4 * (v30 + 1) >= 3 * v21 )
          goto LABEL_24;
        if ( v21 - *(_DWORD *)(a4 + 20) - v26 <= v21 >> 3 )
        {
          sub_CF28B0(a4, v21);
          v32 = *(_DWORD *)(a4 + 24);
          if ( !v32 )
          {
LABEL_64:
            ++*(_DWORD *)(a4 + 16);
            BUG();
          }
          v23 = v41[0];
          v33 = v32 - 1;
          v34 = *(_QWORD *)(a4 + 8);
          v29 = 0;
          v35 = 1;
          v26 = *(_DWORD *)(a4 + 16) + 1;
          v36 = v33 & ((LODWORD(v41[0]) >> 9) ^ (LODWORD(v41[0]) >> 4));
          v17 = (unsigned __int64 *)(v34 + 8LL * v36);
          result = *v17;
          if ( v41[0] != *v17 )
          {
            while ( result != -4096 )
            {
              if ( !v29 && result == -8192 )
                v29 = v17;
              v36 = v33 & (v35 + v36);
              v17 = (unsigned __int64 *)(v34 + 8LL * v36);
              result = *v17;
              if ( v41[0] == *v17 )
                goto LABEL_40;
              ++v35;
            }
            goto LABEL_28;
          }
        }
LABEL_40:
        *(_DWORD *)(a4 + 16) = v26;
        if ( *v17 != -4096 )
          --*(_DWORD *)(a4 + 20);
        *v17 = result;
        v31 = *(_BYTE **)(a5 + 8);
        if ( v31 == *(_BYTE **)(a5 + 16) )
        {
          result = (unsigned __int64)sub_9319A0(a5, v31, v41);
          goto LABEL_21;
        }
        if ( v31 )
        {
          result = v41[0];
          *(_QWORD *)v31 = v41[0];
          v31 = *(_BYTE **)(a5 + 8);
        }
        ++v13;
        *(_QWORD *)(a5 + 8) = v31 + 8;
        if ( v14 == v13 )
        {
LABEL_46:
          a3 = v38;
          v5 = *(_QWORD *)(v5 + 8);
          if ( v40 == v5 )
            return result;
LABEL_12:
          if ( v39 == v5 )
            return result;
          if ( !v5 )
LABEL_14:
            BUG();
          goto LABEL_5;
        }
      }
    }
    ++*(_QWORD *)a4;
LABEL_24:
    sub_CF28B0(a4, 2 * v21);
    v22 = *(_DWORD *)(a4 + 24);
    if ( !v22 )
      goto LABEL_64;
    v23 = v41[0];
    v24 = v22 - 1;
    v25 = *(_QWORD *)(a4 + 8);
    v26 = *(_DWORD *)(a4 + 16) + 1;
    v27 = v24 & ((LODWORD(v41[0]) >> 9) ^ (LODWORD(v41[0]) >> 4));
    v17 = (unsigned __int64 *)(v25 + 8LL * v27);
    result = *v17;
    if ( v41[0] != *v17 )
    {
      v28 = 1;
      v29 = 0;
      while ( result != -4096 )
      {
        if ( result == -8192 && !v29 )
          v29 = v17;
        v27 = v24 & (v28 + v27);
        v17 = (unsigned __int64 *)(v25 + 8LL * v27);
        result = *v17;
        if ( v41[0] == *v17 )
          goto LABEL_40;
        ++v28;
      }
LABEL_28:
      result = v23;
      if ( v29 )
        v17 = v29;
      goto LABEL_40;
    }
    goto LABEL_40;
  }
  return result;
}
