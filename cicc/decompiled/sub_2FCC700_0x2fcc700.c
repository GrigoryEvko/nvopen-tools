// Function: sub_2FCC700
// Address: 0x2fcc700
//
__int64 __fastcall sub_2FCC700(__int64 a1, __int64 a2, __int64 *a3, __int64 *a4)
{
  __int64 result; // rax
  __int64 v8; // r9
  int v9; // esi
  __int64 v10; // rdi
  __int64 v11; // r10
  __int64 v12; // r15
  int v13; // r14d
  unsigned int v14; // ecx
  __int64 *v15; // rdx
  __int64 v16; // r11
  __int64 v17; // r15
  unsigned int v18; // edx
  __int64 *v19; // r8
  int v20; // ecx
  unsigned int v21; // edi
  __int64 v22; // rcx
  char v23; // dl
  __int64 v24; // rdx
  __int64 v25; // rcx
  __int64 v26; // rsi
  int v27; // r15d
  __int64 v28; // rdi
  int v29; // ecx
  unsigned int v30; // edx
  __int64 v31; // r9
  __int64 v32; // rdi
  int v33; // ecx
  unsigned int v34; // edx
  __int64 v35; // r9
  int v36; // r11d
  __int64 *v37; // r10
  int v38; // ecx
  int v39; // ecx
  int v40; // r11d
  __int64 v41; // [rsp+8h] [rbp-38h]
  __int64 v42; // [rsp+8h] [rbp-38h]

  result = a1;
  v8 = *(_QWORD *)a2;
  v9 = *(_BYTE *)(a2 + 8) & 1;
  if ( v9 )
  {
    v10 = *a3;
    v11 = a2 + 16;
    v12 = 512;
    v13 = 15;
    v14 = ((unsigned __int8)((unsigned int)*a3 >> 9) ^ (unsigned __int8)((unsigned int)*a3 >> 4)) & 0xF;
    v15 = (__int64 *)(a2
                    + 16
                    + 32LL
                    * (((unsigned __int8)((unsigned int)*a3 >> 9) ^ (unsigned __int8)((unsigned int)*a3 >> 4)) & 0xF));
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
    v15 = (__int64 *)(v11 + 32LL * v14);
    v16 = *v15;
    if ( v10 == *v15 )
      goto LABEL_6;
  }
  v27 = 1;
  v19 = 0;
  while ( 1 )
  {
    if ( v16 == -4096 )
    {
      if ( !v19 )
        v19 = v15;
      v18 = *(_DWORD *)(a2 + 8);
      *(_QWORD *)a2 = v8 + 1;
      v20 = (v18 >> 1) + 1;
      if ( (_BYTE)v9 )
      {
        v21 = 48;
        LODWORD(v17) = 16;
LABEL_9:
        if ( v21 > 4 * v20 )
        {
          if ( (int)v17 - *(_DWORD *)(a2 + 12) - v20 > (unsigned int)v17 >> 3 )
          {
LABEL_11:
            *(_DWORD *)(a2 + 8) = (2 * (v18 >> 1) + 2) | v18 & 1;
            if ( *v19 != -4096 )
              --*(_DWORD *)(a2 + 12);
            *v19 = *a3;
            v22 = *a4;
            v23 = *((_BYTE *)a4 + 8);
            *((_DWORD *)v19 + 6) = 0;
            v19[1] = v22;
            *((_BYTE *)v19 + 16) = v23;
            if ( (*(_BYTE *)(a2 + 8) & 1) != 0 )
            {
              v24 = a2 + 16;
              v25 = 512;
            }
            else
            {
              v24 = *(_QWORD *)(a2 + 16);
              v25 = 32LL * *(unsigned int *)(a2 + 24);
            }
            v26 = *(_QWORD *)a2;
            *(_QWORD *)result = a2;
            *(_QWORD *)(result + 16) = v19;
            *(_QWORD *)(result + 8) = v26;
            *(_QWORD *)(result + 24) = v25 + v24;
            *(_BYTE *)(result + 32) = 1;
            return result;
          }
          v42 = result;
          sub_2FCC2A0(a2, v17);
          result = v42;
          if ( (*(_BYTE *)(a2 + 8) & 1) != 0 )
          {
            v32 = a2 + 16;
            v33 = 15;
            goto LABEL_29;
          }
          v39 = *(_DWORD *)(a2 + 24);
          v32 = *(_QWORD *)(a2 + 16);
          if ( v39 )
          {
            v33 = v39 - 1;
LABEL_29:
            v34 = v33 & (((unsigned int)*a3 >> 9) ^ ((unsigned int)*a3 >> 4));
            v19 = (__int64 *)(v32 + 32LL * v34);
            v35 = *v19;
            if ( *v19 != *a3 )
            {
              v36 = 1;
              v37 = 0;
              while ( v35 != -4096 )
              {
                if ( v35 == -8192 && !v37 )
                  v37 = v19;
                v34 = v33 & (v36 + v34);
                v19 = (__int64 *)(v32 + 32LL * v34);
                v35 = *v19;
                if ( *a3 == *v19 )
                  goto LABEL_26;
                ++v36;
              }
LABEL_32:
              if ( v37 )
                v19 = v37;
              goto LABEL_26;
            }
            goto LABEL_26;
          }
LABEL_59:
          *(_DWORD *)(a2 + 8) = (2 * (*(_DWORD *)(a2 + 8) >> 1) + 2) | *(_DWORD *)(a2 + 8) & 1;
          BUG();
        }
        v41 = result;
        sub_2FCC2A0(a2, 2 * v17);
        result = v41;
        if ( (*(_BYTE *)(a2 + 8) & 1) != 0 )
        {
          v28 = a2 + 16;
          v29 = 15;
        }
        else
        {
          v38 = *(_DWORD *)(a2 + 24);
          v28 = *(_QWORD *)(a2 + 16);
          if ( !v38 )
            goto LABEL_59;
          v29 = v38 - 1;
        }
        v30 = v29 & (((unsigned int)*a3 >> 9) ^ ((unsigned int)*a3 >> 4));
        v19 = (__int64 *)(v28 + 32LL * v30);
        v31 = *v19;
        if ( *a3 != *v19 )
        {
          v40 = 1;
          v37 = 0;
          while ( v31 != -4096 )
          {
            if ( !v37 && v31 == -8192 )
              v37 = v19;
            v30 = v29 & (v40 + v30);
            v19 = (__int64 *)(v28 + 32LL * v30);
            v31 = *v19;
            if ( *a3 == *v19 )
              goto LABEL_26;
            ++v40;
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
    if ( v16 == -8192 && !v19 )
      v19 = v15;
    v14 = v13 & (v27 + v14);
    v15 = (__int64 *)(v11 + 32LL * v14);
    v16 = *v15;
    if ( v10 == *v15 )
      break;
    ++v27;
  }
  if ( (_BYTE)v9 )
  {
    v12 = 512;
  }
  else
  {
    v17 = *(unsigned int *)(a2 + 24);
LABEL_6:
    v12 = 32 * v17;
  }
LABEL_3:
  *(_QWORD *)result = a2;
  *(_QWORD *)(result + 8) = v8;
  *(_QWORD *)(result + 16) = v15;
  *(_QWORD *)(result + 24) = v11 + v12;
  *(_BYTE *)(result + 32) = 0;
  return result;
}
