// Function: sub_3163490
// Address: 0x3163490
//
__int64 __fastcall sub_3163490(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 result; // rax
  _BYTE *v8; // rdi
  unsigned int v9; // ebx
  __int64 v10; // rbx
  __int64 v11; // r12
  __int64 *v12; // r13
  __int64 v13; // rax
  _QWORD *v14; // rsi
  unsigned __int64 v15; // rax
  int v16; // eax
  _BYTE *v17; // r12
  int v18; // r11d
  __int64 v19; // r8
  unsigned int v20; // edi
  __int64 v21; // rax
  __int64 v22; // rcx
  __int64 v23; // rbx
  __int64 v24; // r13
  unsigned int v25; // esi
  int v26; // esi
  int v27; // esi
  __int64 v28; // r8
  unsigned int v29; // ecx
  int v30; // eax
  __int64 v31; // rdx
  __int64 v32; // rdi
  int v33; // eax
  int v34; // ecx
  int v35; // ecx
  __int64 v36; // rdi
  __int64 v37; // r8
  unsigned int v38; // r14d
  int v39; // r10d
  __int64 v40; // rsi
  size_t v41; // rdx
  int v42; // r14d
  __int64 v43; // [rsp+18h] [rbp-88h]
  int v44; // [rsp+20h] [rbp-80h]
  int v45; // [rsp+20h] [rbp-80h]
  int v46; // [rsp+20h] [rbp-80h]
  __int64 v47; // [rsp+30h] [rbp-70h]
  _BYTE *v48; // [rsp+38h] [rbp-68h]
  _BYTE *v49; // [rsp+40h] [rbp-60h] BYREF
  __int64 v50; // [rsp+48h] [rbp-58h]
  _BYTE v51[80]; // [rsp+50h] [rbp-50h] BYREF

  v47 = *(_QWORD *)*a1;
  result = v47 + 48LL * *(unsigned int *)(*a1 + 8);
  v43 = result;
  if ( result != v47 )
  {
    while ( 1 )
    {
      v8 = v51;
      v50 = 0x400000000LL;
      v49 = v51;
      v9 = *(_DWORD *)(v47 + 8);
      if ( v9 && (_BYTE **)v47 != &v49 )
      {
        v41 = 8LL * v9;
        if ( v9 <= 4
          || (sub_C8D5F0((__int64)&v49, v51, v9, 8u, v9, a6), v8 = v49, (v41 = 8LL * *(unsigned int *)(v47 + 8)) != 0) )
        {
          memcpy(v8, *(const void **)v47, v41);
          v8 = v49;
        }
        LODWORD(v50) = v9;
      }
      v10 = *(_QWORD *)v8;
      v11 = a1[1];
      v12 = *(__int64 **)(*(_QWORD *)v8 + 72LL);
      if ( (unsigned __int8)sub_B4CE70(*(_QWORD *)v8) )
      {
        v13 = *(_QWORD *)(v10 - 32);
        if ( *(_BYTE *)v13 != 17 )
          sub_C64ED0("Coroutines cannot handle non static allocas yet", 1u);
        v14 = *(_QWORD **)(v13 + 24);
        if ( *(_DWORD *)(v13 + 32) > 0x40u )
          v14 = (_QWORD *)*v14;
        v12 = sub_BCD420(v12, (__int64)v14);
      }
      _BitScanReverse64(&v15, 1LL << *(_WORD *)(v10 + 2));
      LOBYTE(v15) = 63 - (v15 ^ 0x3F);
      BYTE1(v15) = 1;
      v16 = sub_315F8A0(v11, (__int64)v12, v15, 0, 0);
      v17 = v49;
      v48 = &v49[8 * (unsigned int)v50];
      if ( v48 != v49 )
        break;
LABEL_20:
      if ( v17 != v51 )
        _libc_free((unsigned __int64)v17);
      v47 += 48;
      result = v47;
      if ( v43 == v47 )
        return result;
    }
    v18 = v16;
    while ( 1 )
    {
      v23 = a1[2];
      v24 = *(_QWORD *)v17;
      v25 = *(_DWORD *)(v23 + 48);
      if ( !v25 )
        break;
      a6 = v25 - 1;
      v19 = *(_QWORD *)(v23 + 32);
      v20 = a6 & (((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4));
      v21 = v19 + 16LL * v20;
      v22 = *(_QWORD *)v21;
      if ( v24 == *(_QWORD *)v21 )
      {
LABEL_11:
        v17 += 8;
        *(_DWORD *)(v21 + 8) = v18;
        if ( v48 == v17 )
          goto LABEL_19;
      }
      else
      {
        v45 = 1;
        v31 = 0;
        while ( v22 != -4096 )
        {
          if ( v22 == -8192 && !v31 )
            v31 = v21;
          v20 = a6 & (v45 + v20);
          v21 = v19 + 16LL * v20;
          v22 = *(_QWORD *)v21;
          if ( v24 == *(_QWORD *)v21 )
            goto LABEL_11;
          ++v45;
        }
        if ( !v31 )
          v31 = v21;
        v33 = *(_DWORD *)(v23 + 40);
        ++*(_QWORD *)(v23 + 24);
        v30 = v33 + 1;
        if ( 4 * v30 < 3 * v25 )
        {
          if ( v25 - *(_DWORD *)(v23 + 44) - v30 <= v25 >> 3 )
          {
            v46 = v18;
            sub_D39D40(v23 + 24, v25);
            v34 = *(_DWORD *)(v23 + 48);
            if ( !v34 )
            {
LABEL_63:
              ++*(_DWORD *)(v23 + 40);
              BUG();
            }
            v35 = v34 - 1;
            v36 = *(_QWORD *)(v23 + 32);
            v37 = 0;
            v38 = v35 & (((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4));
            v18 = v46;
            v39 = 1;
            v30 = *(_DWORD *)(v23 + 40) + 1;
            v31 = v36 + 16LL * v38;
            v40 = *(_QWORD *)v31;
            if ( v24 != *(_QWORD *)v31 )
            {
              while ( v40 != -4096 )
              {
                if ( !v37 && v40 == -8192 )
                  v37 = v31;
                a6 = (unsigned int)(v39 + 1);
                v38 = v35 & (v39 + v38);
                v31 = v36 + 16LL * v38;
                v40 = *(_QWORD *)v31;
                if ( v24 == *(_QWORD *)v31 )
                  goto LABEL_16;
                ++v39;
              }
              if ( v37 )
                v31 = v37;
            }
          }
          goto LABEL_16;
        }
LABEL_14:
        v44 = v18;
        sub_D39D40(v23 + 24, 2 * v25);
        v26 = *(_DWORD *)(v23 + 48);
        if ( !v26 )
          goto LABEL_63;
        v27 = v26 - 1;
        v28 = *(_QWORD *)(v23 + 32);
        v18 = v44;
        v29 = v27 & (((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4));
        v30 = *(_DWORD *)(v23 + 40) + 1;
        v31 = v28 + 16LL * v29;
        v32 = *(_QWORD *)v31;
        if ( v24 != *(_QWORD *)v31 )
        {
          v42 = 1;
          a6 = 0;
          while ( v32 != -4096 )
          {
            if ( !a6 && v32 == -8192 )
              a6 = v31;
            v29 = v27 & (v42 + v29);
            v31 = v28 + 16LL * v29;
            v32 = *(_QWORD *)v31;
            if ( v24 == *(_QWORD *)v31 )
              goto LABEL_16;
            ++v42;
          }
          if ( a6 )
            v31 = a6;
        }
LABEL_16:
        *(_DWORD *)(v23 + 40) = v30;
        if ( *(_QWORD *)v31 != -4096 )
          --*(_DWORD *)(v23 + 44);
        *(_QWORD *)v31 = v24;
        v17 += 8;
        *(_DWORD *)(v31 + 8) = 0;
        *(_DWORD *)(v31 + 8) = v18;
        if ( v48 == v17 )
        {
LABEL_19:
          v17 = v49;
          goto LABEL_20;
        }
      }
    }
    ++*(_QWORD *)(v23 + 24);
    goto LABEL_14;
  }
  return result;
}
