// Function: sub_18EC2C0
// Address: 0x18ec2c0
//
_QWORD *__fastcall sub_18EC2C0(__int64 a1, __int64 a2, __int64 a3, unsigned int a4, __int64 a5)
{
  __int64 v7; // r12
  __int64 v9; // rdi
  _QWORD *result; // rax
  __int64 v11; // r9
  unsigned int v12; // r15d
  unsigned int v13; // esi
  __int64 v14; // r10
  _BYTE *v15; // r8
  __int64 v16; // rax
  __int64 v17; // rdi
  __int64 v18; // rax
  __int64 v19; // rbx
  __int64 v20; // rax
  __int64 v21; // rax
  __int64 v22; // rcx
  int v23; // eax
  int v24; // eax
  __int64 v25; // rbx
  char *v26; // rdi
  int v27; // esi
  int v28; // esi
  __int64 v29; // r8
  unsigned int v30; // edx
  __int64 v31; // rdi
  int v32; // r11d
  __int64 v33; // r10
  int v34; // esi
  int v35; // esi
  __int64 v36; // r8
  int v37; // r11d
  unsigned int v38; // edx
  __int64 v39; // rdi
  unsigned int v40; // [rsp+0h] [rbp-E0h]
  int v41; // [rsp+8h] [rbp-D8h]
  __int64 v42; // [rsp+8h] [rbp-D8h]
  __int64 v43; // [rsp+8h] [rbp-D8h]
  __int64 v44; // [rsp+8h] [rbp-D8h]
  __int64 v45; // [rsp+8h] [rbp-D8h]
  __int64 v46; // [rsp+8h] [rbp-D8h]
  char *v47; // [rsp+10h] [rbp-D0h] BYREF
  __int64 v48; // [rsp+18h] [rbp-C8h]
  _BYTE v49[128]; // [rsp+20h] [rbp-C0h] BYREF
  __int64 v50; // [rsp+A0h] [rbp-40h]
  int v51; // [rsp+A8h] [rbp-38h]

  v7 = a4;
  v9 = *(_QWORD *)a1;
  if ( *(_BYTE *)(a3 + 16) == 78
    && (v21 = *(_QWORD *)(a3 - 24), !*(_BYTE *)(v21 + 16))
    && (*(_BYTE *)(v21 + 33) & 0x20) != 0 )
  {
    result = (_QWORD *)sub_14A3110(v9);
    v11 = a2;
    v12 = (unsigned int)result;
  }
  else
  {
    result = (_QWORD *)sub_14A30E0(v9);
    v11 = a2;
    v12 = (unsigned int)result;
  }
  if ( v12 > 1 )
  {
    v13 = *(_DWORD *)(v11 + 24);
    if ( v13 )
    {
      v14 = *(_QWORD *)(v11 + 8);
      LODWORD(v15) = (v13 - 1) & (((unsigned int)a5 >> 9) ^ ((unsigned int)a5 >> 4));
      v16 = v14 + 16LL * (unsigned int)v15;
      v17 = *(_QWORD *)v16;
      if ( a5 == *(_QWORD *)v16 )
      {
LABEL_6:
        v18 = *(unsigned int *)(v16 + 8);
LABEL_7:
        v19 = *(_QWORD *)(a1 + 32) + 160 * v18;
        *(_DWORD *)(v19 + 152) += v12;
        v20 = *(unsigned int *)(v19 + 8);
        if ( (unsigned int)v20 >= *(_DWORD *)(v19 + 12) )
        {
          sub_16CD150(v19, (const void *)(v19 + 16), 0, 16, (int)v15, v11);
          v20 = *(unsigned int *)(v19 + 8);
        }
        result = (_QWORD *)(*(_QWORD *)v19 + 16 * v20);
        *result = a3;
        result[1] = v7;
        ++*(_DWORD *)(v19 + 8);
        return result;
      }
      v41 = 1;
      v22 = 0;
      while ( v17 != -8 )
      {
        if ( !v22 && v17 == -16 )
          v22 = v16;
        LODWORD(v15) = (v13 - 1) & (v41 + (_DWORD)v15);
        v16 = v14 + 16LL * (unsigned int)v15;
        v17 = *(_QWORD *)v16;
        if ( a5 == *(_QWORD *)v16 )
          goto LABEL_6;
        ++v41;
      }
      if ( !v22 )
        v22 = v16;
      v23 = *(_DWORD *)(v11 + 16);
      ++*(_QWORD *)v11;
      v24 = v23 + 1;
      if ( 4 * v24 < 3 * v13 )
      {
        if ( v13 - *(_DWORD *)(v11 + 20) - v24 > v13 >> 3 )
        {
LABEL_19:
          *(_DWORD *)(v11 + 16) = v24;
          if ( *(_QWORD *)v22 != -8 )
            --*(_DWORD *)(v11 + 20);
          v15 = v49;
          *(_QWORD *)v22 = a5;
          *(_DWORD *)(v22 + 8) = 0;
          v47 = v49;
          v50 = a5;
          v25 = *(_QWORD *)(a1 + 40);
          v48 = 0x800000000LL;
          v51 = 0;
          if ( v25 == *(_QWORD *)(a1 + 48) )
          {
            v43 = v22;
            sub_18EBE40((unsigned __int64 **)(a1 + 32), v25, (__int64)&v47, v22);
            v26 = v47;
            v15 = v49;
            v22 = v43;
          }
          else
          {
            v26 = v49;
            if ( v25 )
            {
              *(_QWORD *)(v25 + 8) = 0x800000000LL;
              *(_QWORD *)v25 = v25 + 16;
              if ( (_DWORD)v48 )
              {
                v44 = v22;
                sub_18E63F0(v25, &v47, (__int64)&v47, v22, (int)v49, v11);
                v15 = v49;
                v22 = v44;
              }
              v26 = v47;
              *(_QWORD *)(v25 + 144) = v50;
              *(_DWORD *)(v25 + 152) = v51;
              v25 = *(_QWORD *)(a1 + 40);
            }
            *(_QWORD *)(a1 + 40) = v25 + 160;
          }
          if ( v26 != v49 )
          {
            v42 = v22;
            _libc_free((unsigned __int64)v26);
            v22 = v42;
          }
          v18 = -858993459 * (unsigned int)((__int64)(*(_QWORD *)(a1 + 40) - *(_QWORD *)(a1 + 32)) >> 5) - 1;
          *(_DWORD *)(v22 + 8) = v18;
          goto LABEL_7;
        }
        v46 = v11;
        v40 = ((unsigned int)a5 >> 9) ^ ((unsigned int)a5 >> 4);
        sub_18EC100(v11, v13);
        v11 = v46;
        v34 = *(_DWORD *)(v46 + 24);
        if ( v34 )
        {
          v35 = v34 - 1;
          v36 = *(_QWORD *)(v46 + 8);
          v33 = 0;
          v37 = 1;
          v38 = v35 & v40;
          v24 = *(_DWORD *)(v46 + 16) + 1;
          v22 = v36 + 16LL * (v35 & v40);
          v39 = *(_QWORD *)v22;
          if ( a5 == *(_QWORD *)v22 )
            goto LABEL_19;
          while ( v39 != -8 )
          {
            if ( !v33 && v39 == -16 )
              v33 = v22;
            v38 = v35 & (v37 + v38);
            v22 = v36 + 16LL * v38;
            v39 = *(_QWORD *)v22;
            if ( a5 == *(_QWORD *)v22 )
              goto LABEL_19;
            ++v37;
          }
          goto LABEL_36;
        }
        goto LABEL_57;
      }
    }
    else
    {
      ++*(_QWORD *)v11;
    }
    v45 = v11;
    sub_18EC100(v11, 2 * v13);
    v11 = v45;
    v27 = *(_DWORD *)(v45 + 24);
    if ( v27 )
    {
      v28 = v27 - 1;
      v29 = *(_QWORD *)(v45 + 8);
      v30 = v28 & (((unsigned int)a5 >> 9) ^ ((unsigned int)a5 >> 4));
      v24 = *(_DWORD *)(v45 + 16) + 1;
      v22 = v29 + 16LL * v30;
      v31 = *(_QWORD *)v22;
      if ( a5 == *(_QWORD *)v22 )
        goto LABEL_19;
      v32 = 1;
      v33 = 0;
      while ( v31 != -8 )
      {
        if ( v31 == -16 && !v33 )
          v33 = v22;
        v30 = v28 & (v32 + v30);
        v22 = v29 + 16LL * v30;
        v31 = *(_QWORD *)v22;
        if ( a5 == *(_QWORD *)v22 )
          goto LABEL_19;
        ++v32;
      }
LABEL_36:
      if ( v33 )
        v22 = v33;
      goto LABEL_19;
    }
LABEL_57:
    ++*(_DWORD *)(v11 + 16);
    BUG();
  }
  return result;
}
