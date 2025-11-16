// Function: sub_2047830
// Address: 0x2047830
//
__int64 __fastcall sub_2047830(__int64 a1, __int64 a2, __int64 *a3, _DWORD *a4)
{
  __int64 result; // rax
  __int64 v8; // r8
  int v9; // esi
  __int64 v10; // rdi
  __int64 v11; // r9
  __int64 v12; // r15
  int v13; // r14d
  unsigned int v14; // ecx
  __int64 *v15; // rdx
  __int64 v16; // r11
  __int64 v17; // r15
  unsigned int v18; // edx
  __int64 *v19; // r10
  int v20; // ecx
  unsigned int v21; // edi
  __int64 v22; // rdx
  __int64 v23; // rcx
  __int64 v24; // rsi
  int v25; // r15d
  __int64 v26; // rdi
  int v27; // ecx
  unsigned int v28; // edx
  __int64 v29; // r8
  __int64 v30; // rdi
  int v31; // ecx
  unsigned int v32; // edx
  __int64 v33; // r8
  int v34; // r11d
  __int64 *v35; // r9
  int v36; // ecx
  int v37; // ecx
  int v38; // r11d
  __int64 v39; // [rsp+8h] [rbp-38h]
  __int64 v40; // [rsp+8h] [rbp-38h]

  result = a1;
  v8 = *(_QWORD *)a2;
  v9 = *(_BYTE *)(a2 + 8) & 1;
  if ( v9 )
  {
    v10 = *a3;
    v11 = a2 + 16;
    v12 = 128;
    v13 = 7;
    v14 = ((unsigned __int8)((unsigned int)*a3 >> 9) ^ (unsigned __int8)((unsigned int)*a3 >> 4)) & 7;
    v15 = (__int64 *)(a2
                    + 16
                    + 16LL
                    * (((unsigned __int8)((unsigned int)*a3 >> 9) ^ (unsigned __int8)((unsigned int)*a3 >> 4)) & 7));
    v16 = *v15;
    if ( v10 == *v15 )
      goto LABEL_3;
  }
  else
  {
    v17 = *(unsigned int *)(a2 + 24);
    if ( !(_DWORD)v17 )
    {
      v18 = *(_DWORD *)(a2 + 8);
      v19 = 0;
      *(_QWORD *)a2 = v8 + 1;
      v20 = (v18 >> 1) + 1;
      goto LABEL_8;
    }
    v10 = *a3;
    v13 = v17 - 1;
    v11 = *(_QWORD *)(a2 + 16);
    v14 = (v17 - 1) & (((unsigned int)*a3 >> 4) ^ ((unsigned int)*a3 >> 9));
    v15 = (__int64 *)(v11 + 16LL * v14);
    v16 = *v15;
    if ( v10 == *v15 )
      goto LABEL_6;
  }
  v25 = 1;
  v19 = 0;
  while ( 1 )
  {
    if ( v16 == -8 )
    {
      if ( !v19 )
        v19 = v15;
      v18 = *(_DWORD *)(a2 + 8);
      *(_QWORD *)a2 = v8 + 1;
      v20 = (v18 >> 1) + 1;
      if ( (_BYTE)v9 )
      {
        v21 = 24;
        LODWORD(v17) = 8;
LABEL_9:
        if ( 4 * v20 < v21 )
        {
          if ( (int)v17 - *(_DWORD *)(a2 + 12) - v20 > (unsigned int)v17 >> 3 )
          {
LABEL_11:
            *(_DWORD *)(a2 + 8) = (2 * (v18 >> 1) + 2) | v18 & 1;
            if ( *v19 != -8 )
              --*(_DWORD *)(a2 + 12);
            *v19 = *a3;
            *((_DWORD *)v19 + 2) = *a4;
            if ( (*(_BYTE *)(a2 + 8) & 1) != 0 )
            {
              v22 = a2 + 16;
              v23 = 128;
            }
            else
            {
              v22 = *(_QWORD *)(a2 + 16);
              v23 = 16LL * *(unsigned int *)(a2 + 24);
            }
            v24 = *(_QWORD *)a2;
            *(_QWORD *)result = a2;
            *(_QWORD *)(result + 16) = v19;
            *(_QWORD *)(result + 8) = v24;
            *(_QWORD *)(result + 24) = v23 + v22;
            *(_BYTE *)(result + 32) = 1;
            return result;
          }
          v40 = result;
          sub_2046A80(a2, v17);
          result = v40;
          if ( (*(_BYTE *)(a2 + 8) & 1) != 0 )
          {
            v30 = a2 + 16;
            v31 = 7;
            goto LABEL_29;
          }
          v37 = *(_DWORD *)(a2 + 24);
          v30 = *(_QWORD *)(a2 + 16);
          if ( v37 )
          {
            v31 = v37 - 1;
LABEL_29:
            v32 = v31 & (((unsigned int)*a3 >> 9) ^ ((unsigned int)*a3 >> 4));
            v19 = (__int64 *)(v30 + 16LL * v32);
            v33 = *v19;
            if ( *v19 != *a3 )
            {
              v34 = 1;
              v35 = 0;
              while ( v33 != -8 )
              {
                if ( v33 == -16 && !v35 )
                  v35 = v19;
                v32 = v31 & (v34 + v32);
                v19 = (__int64 *)(v30 + 16LL * v32);
                v33 = *v19;
                if ( *a3 == *v19 )
                  goto LABEL_26;
                ++v34;
              }
LABEL_32:
              if ( v35 )
                v19 = v35;
              goto LABEL_26;
            }
            goto LABEL_26;
          }
LABEL_59:
          *(_DWORD *)(a2 + 8) = (2 * (*(_DWORD *)(a2 + 8) >> 1) + 2) | *(_DWORD *)(a2 + 8) & 1;
          BUG();
        }
        v39 = result;
        sub_2046A80(a2, 2 * v17);
        result = v39;
        if ( (*(_BYTE *)(a2 + 8) & 1) != 0 )
        {
          v26 = a2 + 16;
          v27 = 7;
        }
        else
        {
          v36 = *(_DWORD *)(a2 + 24);
          v26 = *(_QWORD *)(a2 + 16);
          if ( !v36 )
            goto LABEL_59;
          v27 = v36 - 1;
        }
        v28 = v27 & (((unsigned int)*a3 >> 9) ^ ((unsigned int)*a3 >> 4));
        v19 = (__int64 *)(v26 + 16LL * v28);
        v29 = *v19;
        if ( *a3 != *v19 )
        {
          v38 = 1;
          v35 = 0;
          while ( v29 != -8 )
          {
            if ( !v35 && v29 == -16 )
              v35 = v19;
            v28 = v27 & (v38 + v28);
            v19 = (__int64 *)(v26 + 16LL * v28);
            v29 = *v19;
            if ( *a3 == *v19 )
              goto LABEL_26;
            ++v38;
          }
          goto LABEL_32;
        }
LABEL_26:
        v18 = *(_DWORD *)(a2 + 8);
        goto LABEL_11;
      }
      LODWORD(v17) = *(_DWORD *)(a2 + 24);
LABEL_8:
      v21 = 3 * v17;
      goto LABEL_9;
    }
    if ( !v19 && v16 == -16 )
      v19 = v15;
    v14 = v13 & (v25 + v14);
    v15 = (__int64 *)(v11 + 16LL * v14);
    v16 = *v15;
    if ( *v15 == v10 )
      break;
    ++v25;
  }
  if ( (_BYTE)v9 )
  {
    v12 = 128;
  }
  else
  {
    v17 = *(unsigned int *)(a2 + 24);
LABEL_6:
    v12 = 16 * v17;
  }
LABEL_3:
  *(_QWORD *)result = a2;
  *(_QWORD *)(result + 8) = v8;
  *(_QWORD *)(result + 16) = v15;
  *(_QWORD *)(result + 24) = v11 + v12;
  *(_BYTE *)(result + 32) = 0;
  return result;
}
