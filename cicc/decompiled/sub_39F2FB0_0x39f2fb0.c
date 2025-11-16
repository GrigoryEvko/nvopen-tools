// Function: sub_39F2FB0
// Address: 0x39f2fb0
//
__int64 __fastcall sub_39F2FB0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 result; // rax
  unsigned int v6; // esi
  __int64 v7; // rdi
  unsigned int v8; // ecx
  __int64 v9; // r9
  __int64 v10; // r13
  unsigned int v11; // esi
  __int64 v12; // rdi
  unsigned int v13; // ecx
  __int64 v14; // rdx
  int v15; // r11d
  __int64 v16; // rdx
  int v17; // eax
  int v18; // ecx
  int v19; // eax
  int v20; // esi
  __int64 v21; // r8
  __int64 v22; // rdi
  int v23; // r10d
  __int64 v24; // r9
  int v25; // eax
  int v26; // r9d
  __int64 v27; // r8
  __int64 v28; // rdi
  unsigned int v29; // r13d
  __int64 v30; // rsi
  __int64 v31; // r9
  int v32; // r10d
  int v33; // ecx
  int v34; // ecx
  int v35; // ecx
  int v36; // ecx
  int v37; // esi
  unsigned int j; // edx
  __int64 v39; // rdi
  int v40; // ecx
  int v41; // ecx
  __int64 v42; // r8
  int v43; // esi
  unsigned int i; // r15d
  __int64 *v45; // rdx
  __int64 v46; // rdi
  unsigned int v47; // r15d
  unsigned int v48; // edx

  result = sub_38D58D0(a1, a2, a3);
  if ( !*(_BYTE *)(a2 + 167) )
  {
    result = strlen((const char *)(a2 + 152));
    if ( result == 7
      && *(_DWORD *)(a2 + 152) == 1464098655
      && *(_WORD *)(a2 + 156) == 21057
      && *(_BYTE *)(a2 + 158) == 70 )
    {
      *(_BYTE *)(a1 + 322) = 1;
    }
  }
  if ( *(_BYTE *)(a1 + 320) )
  {
    v6 = *(_DWORD *)(a1 + 352);
    if ( v6 )
    {
      v7 = *(_QWORD *)(a1 + 336);
      v8 = (v6 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      result = v7 + 16LL * v8;
      v9 = *(_QWORD *)result;
      if ( *(_QWORD *)result == a2 )
      {
LABEL_8:
        if ( *(_BYTE *)(result + 8) )
          return result;
        v10 = *(_QWORD *)(a2 + 8);
        if ( v10 )
          return result;
        goto LABEL_10;
      }
      v15 = 1;
      v16 = 0;
      while ( v9 != -8 )
      {
        if ( !v16 && v9 == -16 )
          v16 = result;
        v8 = (v6 - 1) & (v15 + v8);
        result = v7 + 16LL * v8;
        v9 = *(_QWORD *)result;
        if ( *(_QWORD *)result == a2 )
          goto LABEL_8;
        ++v15;
      }
      if ( !v16 )
        v16 = result;
      v17 = *(_DWORD *)(a1 + 344);
      ++*(_QWORD *)(a1 + 328);
      v18 = v17 + 1;
      if ( 4 * (v17 + 1) < 3 * v6 )
      {
        result = v6 - *(_DWORD *)(a1 + 348) - v18;
        if ( (unsigned int)result > v6 >> 3 )
          goto LABEL_22;
        sub_39F2DF0(a1 + 328, v6);
        v25 = *(_DWORD *)(a1 + 352);
        if ( v25 )
        {
          result = (unsigned int)(v25 - 1);
          v26 = 1;
          v27 = 0;
          v28 = *(_QWORD *)(a1 + 336);
          v29 = result & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
          v18 = *(_DWORD *)(a1 + 344) + 1;
          v16 = v28 + 16LL * v29;
          v30 = *(_QWORD *)v16;
          if ( *(_QWORD *)v16 != a2 )
          {
            while ( v30 != -8 )
            {
              if ( v30 == -16 && !v27 )
                v27 = v16;
              v29 = result & (v26 + v29);
              v16 = v28 + 16LL * v29;
              v30 = *(_QWORD *)v16;
              if ( *(_QWORD *)v16 == a2 )
                goto LABEL_22;
              ++v26;
            }
            if ( v27 )
              v16 = v27;
          }
          goto LABEL_22;
        }
        goto LABEL_95;
      }
    }
    else
    {
      ++*(_QWORD *)(a1 + 328);
    }
    sub_39F2DF0(a1 + 328, 2 * v6);
    v19 = *(_DWORD *)(a1 + 352);
    if ( v19 )
    {
      v20 = v19 - 1;
      v21 = *(_QWORD *)(a1 + 336);
      result = (v19 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v18 = *(_DWORD *)(a1 + 344) + 1;
      v16 = v21 + 16 * result;
      v22 = *(_QWORD *)v16;
      if ( *(_QWORD *)v16 != a2 )
      {
        v23 = 1;
        v24 = 0;
        while ( v22 != -8 )
        {
          if ( !v24 && v22 == -16 )
            v24 = v16;
          result = v20 & (unsigned int)(v23 + result);
          v16 = v21 + 16LL * (unsigned int)result;
          v22 = *(_QWORD *)v16;
          if ( *(_QWORD *)v16 == a2 )
            goto LABEL_22;
          ++v23;
        }
        if ( v24 )
          v16 = v24;
      }
LABEL_22:
      *(_DWORD *)(a1 + 344) = v18;
      if ( *(_QWORD *)v16 != -8 )
        --*(_DWORD *)(a1 + 348);
      *(_QWORD *)v16 = a2;
      *(_BYTE *)(v16 + 8) = 0;
      v10 = *(_QWORD *)(a2 + 8);
      if ( v10 )
        return result;
LABEL_10:
      *(_QWORD *)(a2 + 8) = sub_38BFE40(*(_QWORD *)(a1 + 8));
      v11 = *(_DWORD *)(a1 + 352);
      if ( v11 )
      {
        v12 = *(_QWORD *)(a1 + 336);
        v13 = (v11 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
        result = v12 + 16LL * v13;
        v14 = *(_QWORD *)result;
        if ( *(_QWORD *)result == a2 )
        {
LABEL_12:
          *(_BYTE *)(result + 8) = 1;
          return result;
        }
        v31 = 0;
        v32 = 1;
        while ( v14 != -8 )
        {
          if ( !v31 && v14 == -16 )
            v31 = result;
          v13 = (v11 - 1) & (v32 + v13);
          result = v12 + 16LL * v13;
          v14 = *(_QWORD *)result;
          if ( *(_QWORD *)result == a2 )
            goto LABEL_12;
          ++v32;
        }
        v33 = *(_DWORD *)(a1 + 344);
        if ( v31 )
          result = v31;
        ++*(_QWORD *)(a1 + 328);
        v34 = v33 + 1;
        if ( 4 * v34 < 3 * v11 )
        {
          if ( v11 - *(_DWORD *)(a1 + 348) - v34 > v11 >> 3 )
          {
LABEL_51:
            *(_DWORD *)(a1 + 344) = v34;
            if ( *(_QWORD *)result != -8 )
              --*(_DWORD *)(a1 + 348);
            *(_QWORD *)result = a2;
            *(_BYTE *)(result + 8) = 0;
            goto LABEL_12;
          }
          sub_39F2DF0(a1 + 328, v11);
          v40 = *(_DWORD *)(a1 + 352);
          if ( v40 )
          {
            v41 = v40 - 1;
            result = 0;
            v43 = 1;
            for ( i = v41 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)); ; i = v41 & v47 )
            {
              v42 = *(_QWORD *)(a1 + 336);
              v45 = (__int64 *)(v42 + 16LL * i);
              v46 = *v45;
              if ( *v45 == a2 )
              {
                v34 = *(_DWORD *)(a1 + 344) + 1;
                result = v42 + 16LL * i;
                goto LABEL_51;
              }
              if ( v46 == -8 )
                break;
              if ( v46 != -16 || result )
                v45 = (__int64 *)result;
              v47 = v43 + i;
              result = (__int64)v45;
              ++v43;
            }
            v34 = *(_DWORD *)(a1 + 344) + 1;
            if ( !result )
              result = v42 + 16LL * i;
            goto LABEL_51;
          }
LABEL_94:
          ++*(_DWORD *)(a1 + 344);
          BUG();
        }
      }
      else
      {
        ++*(_QWORD *)(a1 + 328);
      }
      sub_39F2DF0(a1 + 328, 2 * v11);
      v35 = *(_DWORD *)(a1 + 352);
      if ( v35 )
      {
        v36 = v35 - 1;
        v37 = 1;
        for ( j = v36 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)); ; j = v36 & v48 )
        {
          result = *(_QWORD *)(a1 + 336) + 16LL * j;
          v39 = *(_QWORD *)result;
          if ( *(_QWORD *)result == a2 )
          {
            v34 = *(_DWORD *)(a1 + 344) + 1;
            goto LABEL_51;
          }
          if ( v39 == -8 )
            break;
          if ( v10 || v39 != -16 )
            result = v10;
          v48 = v37 + j;
          v10 = result;
          ++v37;
        }
        v34 = *(_DWORD *)(a1 + 344) + 1;
        if ( v10 )
          result = v10;
        goto LABEL_51;
      }
      goto LABEL_94;
    }
LABEL_95:
    ++*(_DWORD *)(a1 + 344);
    BUG();
  }
  return result;
}
