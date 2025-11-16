// Function: sub_E3A5B0
// Address: 0xe3a5b0
//
__int64 __fastcall sub_E3A5B0(__int64 a1, __int64 a2)
{
  _QWORD *v2; // r15
  int v3; // r14d
  __int64 v4; // r12
  __int64 result; // rax
  int v7; // r10d
  __int64 v8; // r9
  unsigned int v9; // r8d
  char *v10; // rdx
  __int64 v11; // rdi
  __int64 v12; // rsi
  int v13; // ecx
  int v14; // eax
  int v15; // esi
  __int64 v16; // rdi
  unsigned int v17; // eax
  int v18; // edx
  __int64 v19; // rcx
  __int64 v20; // r8
  int v21; // r10d
  int v22; // r11d
  __int64 v23; // r11
  char *v24; // rcx
  __int64 v25; // r11
  int v26; // edx
  int v27; // eax
  unsigned __int64 v28; // rdx
  int v29; // r13d
  int v30; // eax
  _DWORD *v31; // rdx
  unsigned __int64 v32; // rax
  __int64 v33; // r13
  unsigned int v34; // eax
  __int64 v35; // r8
  __int64 v36; // r8
  __int64 v37; // r9
  __int64 v38; // rax
  int v39; // eax
  int v40; // eax
  __int64 v41; // rdi
  unsigned int v42; // r13d
  __int64 v43; // r8
  __int64 v44; // rsi
  int v45; // r11d
  char *v46; // [rsp+8h] [rbp-E8h]
  int v47; // [rsp+10h] [rbp-E0h]
  unsigned int v48; // [rsp+14h] [rbp-DCh]
  __int64 v49; // [rsp+18h] [rbp-D8h]
  unsigned __int64 v50; // [rsp+30h] [rbp-C0h]
  __int64 v51; // [rsp+38h] [rbp-B8h]
  __int64 *v52; // [rsp+40h] [rbp-B0h] BYREF
  __int64 v53; // [rsp+48h] [rbp-A8h]
  _BYTE v54[32]; // [rsp+50h] [rbp-A0h] BYREF
  _QWORD *v55; // [rsp+70h] [rbp-80h] BYREF
  __int64 v56; // [rsp+78h] [rbp-78h]
  _QWORD v57[14]; // [rsp+80h] [rbp-70h] BYREF

  v2 = v57;
  v3 = 0;
  v4 = a2;
  v52 = (__int64 *)v54;
  v53 = 0x800000000LL;
  v56 = 0x800000001LL;
  v49 = a1 + 8;
  LODWORD(result) = 1;
  v55 = v57;
  v57[0] = a2;
  while ( 2 )
  {
    v12 = *(unsigned int *)(a1 + 32);
    if ( !(_DWORD)v12 )
    {
      ++*(_QWORD *)(a1 + 8);
      goto LABEL_9;
    }
    v7 = v12 - 1;
    v8 = *(_QWORD *)(a1 + 16);
    v9 = (v12 - 1) & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
    v10 = (char *)(v8 + 16LL * v9);
    v11 = *(_QWORD *)v10;
    if ( *(_QWORD *)v10 == v4 )
    {
      v12 = (__int64)v52;
      v13 = v53;
      if ( (_DWORD)result == *((_DWORD *)v52 + (unsigned int)v53 - 1) )
        goto LABEL_23;
      goto LABEL_4;
    }
    v48 = (v12 - 1) & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
    v25 = *(_QWORD *)v10;
    v19 = 0;
    v47 = 1;
    while ( 1 )
    {
      if ( v25 == -4096 )
      {
        v27 = *(_DWORD *)(a1 + 24);
        if ( !v19 )
          v19 = (__int64)v10;
        ++*(_QWORD *)(a1 + 8);
        v18 = v27 + 1;
        if ( 4 * (v27 + 1) < (unsigned int)(3 * v12) )
        {
          if ( (int)v12 - *(_DWORD *)(a1 + 28) - v18 > (unsigned int)v12 >> 3 )
          {
LABEL_35:
            *(_DWORD *)(a1 + 24) = v18;
            if ( *(_QWORD *)v19 != -4096 )
              --*(_DWORD *)(a1 + 28);
            ++v3;
            *(_QWORD *)v19 = v4;
            *(_DWORD *)(v19 + 8) = v3;
            v28 = (unsigned int)v53;
            *(_DWORD *)(v19 + 12) = 0;
            v29 = v56;
            v30 = v28;
            if ( v28 >= HIDWORD(v53) )
            {
              if ( HIDWORD(v53) < v28 + 1 )
              {
                sub_C8D5F0((__int64)&v52, v54, v28 + 1, 4u, v28 + 1, v8);
                v28 = (unsigned int)v53;
              }
              *((_DWORD *)v52 + v28) = v29;
              LODWORD(v53) = v53 + 1;
            }
            else
            {
              v31 = (_DWORD *)v52 + v28;
              if ( v31 )
              {
                *v31 = v56;
                v30 = v53;
              }
              LODWORD(v53) = v30 + 1;
            }
            v32 = *(_QWORD *)(v4 + 48) & 0xFFFFFFFFFFFFFFF8LL;
            if ( v32 == v4 + 48 )
              goto LABEL_48;
            if ( !v32 )
              BUG();
            v33 = v32 - 24;
            if ( (unsigned int)*(unsigned __int8 *)(v32 - 24) - 30 > 0xA )
            {
LABEL_48:
              v34 = 0;
              v35 = 0;
              v33 = 0;
            }
            else
            {
              v34 = sub_B46E30(v33);
              v35 = v33;
            }
            v51 &= 0xFFFFFFFF00000000LL;
            v50 = v34 | v50 & 0xFFFFFFFF00000000LL;
            v12 = (__int64)&v55[(unsigned int)v56];
            sub_E37DE0(&v55, (__int64 *)v12, v33, v51, v35, v50);
            v38 = *(unsigned int *)(a1 + 48);
            if ( v38 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 52) )
            {
              v12 = a1 + 56;
              sub_C8D5F0(a1 + 40, (const void *)(a1 + 56), v38 + 1, 8u, v36, v37);
              v38 = *(unsigned int *)(a1 + 48);
            }
            *(_QWORD *)(*(_QWORD *)(a1 + 40) + 8 * v38) = v4;
            result = (unsigned int)v56;
            ++*(_DWORD *)(a1 + 48);
            v2 = v55;
            goto LABEL_5;
          }
          sub_E3A3D0(v49, v12);
          v39 = *(_DWORD *)(a1 + 32);
          if ( v39 )
          {
            v40 = v39 - 1;
            v41 = *(_QWORD *)(a1 + 16);
            v8 = 1;
            v42 = v40 & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
            v43 = 0;
            v18 = *(_DWORD *)(a1 + 24) + 1;
            v19 = v41 + 16LL * v42;
            v44 = *(_QWORD *)v19;
            if ( *(_QWORD *)v19 != v4 )
            {
              while ( v44 != -4096 )
              {
                if ( !v43 && v44 == -8192 )
                  v43 = v19;
                v42 = v40 & (v8 + v42);
                v19 = v41 + 16LL * v42;
                v44 = *(_QWORD *)v19;
                if ( *(_QWORD *)v19 == v4 )
                  goto LABEL_35;
                v8 = (unsigned int)(v8 + 1);
              }
              if ( v43 )
                v19 = v43;
            }
            goto LABEL_35;
          }
LABEL_73:
          ++*(_DWORD *)(a1 + 24);
          BUG();
        }
LABEL_9:
        sub_E3A3D0(v49, 2 * v12);
        v14 = *(_DWORD *)(a1 + 32);
        if ( v14 )
        {
          v15 = v14 - 1;
          v16 = *(_QWORD *)(a1 + 16);
          v17 = (v14 - 1) & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
          v18 = *(_DWORD *)(a1 + 24) + 1;
          v19 = v16 + 16LL * v17;
          v20 = *(_QWORD *)v19;
          if ( *(_QWORD *)v19 != v4 )
          {
            v21 = 1;
            v8 = 0;
            while ( v20 != -4096 )
            {
              if ( v20 == -8192 && !v8 )
                v8 = v19;
              v17 = v15 & (v21 + v17);
              v19 = v16 + 16LL * v17;
              v20 = *(_QWORD *)v19;
              if ( *(_QWORD *)v19 == v4 )
                goto LABEL_35;
              ++v21;
            }
            if ( v8 )
              v19 = v8;
          }
          goto LABEL_35;
        }
        goto LABEL_73;
      }
      if ( v25 != -8192 || v19 )
        v10 = (char *)v19;
      v22 = v47++;
      v23 = v7 & (v22 + v48);
      v48 = v23;
      v24 = (char *)(v8 + 16 * v23);
      v25 = *(_QWORD *)v24;
      v46 = v24;
      if ( *(_QWORD *)v24 == v4 )
        break;
      v19 = (__int64)v10;
      v10 = v46;
    }
    v13 = v53;
    if ( (_DWORD)result == *((_DWORD *)v52 + (unsigned int)v53 - 1) )
    {
      v26 = 1;
      while ( v11 != -4096 )
      {
        v45 = v26 + 1;
        v9 = v7 & (v26 + v9);
        v10 = (char *)(v8 + 16LL * v9);
        v11 = *(_QWORD *)v10;
        if ( *(_QWORD *)v10 == v4 )
          goto LABEL_23;
        v26 = v45;
      }
      v12 *= 16;
      v10 = (char *)(v8 + v12);
LABEL_23:
      *((_DWORD *)v10 + 3) = v3;
      LODWORD(v53) = v13 - 1;
    }
LABEL_4:
    result = (unsigned int)(result - 1);
    LODWORD(v56) = result;
LABEL_5:
    if ( (_DWORD)result )
    {
      v4 = v2[(unsigned int)result - 1];
      continue;
    }
    break;
  }
  if ( v2 != v57 )
    result = _libc_free(v2, v12);
  if ( v52 != (__int64 *)v54 )
    return _libc_free(v52, v12);
  return result;
}
