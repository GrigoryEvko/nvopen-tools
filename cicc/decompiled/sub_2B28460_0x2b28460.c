// Function: sub_2B28460
// Address: 0x2b28460
//
_BYTE *__fastcall sub_2B28460(__int64 a1, __int64 a2)
{
  __int64 v2; // r13
  __int64 v3; // r9
  _BYTE *v4; // r11
  __int64 v5; // rax
  __int64 v6; // r14
  unsigned __int64 v7; // rdx
  __int64 v8; // rax
  unsigned int v9; // r12d
  __int64 v10; // r10
  unsigned __int64 v11; // rbx
  _QWORD *v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // r8
  __int64 v17; // rcx
  __int64 v18; // rsi
  __int64 v19; // r8
  _BYTE *result; // rax
  __int64 v21; // rcx
  __int64 v22; // rax
  __int64 v23; // rcx
  unsigned __int64 v24; // rdx
  __int64 v25; // rdx
  _BYTE *v26; // [rsp+10h] [rbp-180h]
  _BYTE *v27; // [rsp+18h] [rbp-178h]
  __int64 v28; // [rsp+20h] [rbp-170h]
  __int64 v29; // [rsp+20h] [rbp-170h]
  _BYTE *v30; // [rsp+30h] [rbp-160h] BYREF
  __int64 v31; // [rsp+38h] [rbp-158h]
  _BYTE v32[128]; // [rsp+40h] [rbp-150h] BYREF
  int v33; // [rsp+C0h] [rbp-D0h]
  char v34; // [rsp+C4h] [rbp-CCh]
  _BYTE *v35; // [rsp+C8h] [rbp-C8h] BYREF
  __int64 v36; // [rsp+D0h] [rbp-C0h]
  _BYTE v37[184]; // [rsp+D8h] [rbp-B8h] BYREF

  v3 = a2;
  v4 = v32;
  v5 = *(_QWORD *)(a1 + 8);
  v30 = v32;
  v6 = *(_QWORD *)(v5 + 80);
  v31 = 0x800000000LL;
  if ( *(_DWORD *)(v6 + 12) == 1 )
  {
    v34 = 0;
    v33 = 1;
    v35 = v37;
    v36 = 0x800000000LL;
  }
  else
  {
    v7 = 8;
    v8 = 0;
    v9 = 0;
    while ( 1 )
    {
      v10 = v9;
      v11 = v2 & 0xFFFFFF0000000000LL;
      v2 &= 0xFFFFFF0000000000LL;
      if ( v8 + 1 > v7 )
      {
        v26 = v4;
        sub_C8D5F0((__int64)&v30, v4, v8 + 1, 0x10u, 0xFFFFFFFF00000000LL, v8 + 1);
        v8 = (unsigned int)v31;
        v10 = v9;
        v4 = v26;
      }
      v12 = &v30[16 * v8];
      ++v9;
      *v12 = v10;
      v12[1] = v11;
      v8 = (unsigned int)(v31 + 1);
      v13 = (unsigned int)(*(_DWORD *)(v6 + 12) - 1);
      LODWORD(v31) = v31 + 1;
      if ( (unsigned int)v13 <= v9 )
        break;
      v7 = HIDWORD(v31);
    }
    v34 = 0;
    v3 = a2;
    v33 = 1;
    v35 = v37;
    v36 = 0x800000000LL;
    if ( (_DWORD)v8 )
    {
      v27 = v4;
      sub_2B0D350((__int64)&v35, (__int64)&v30, v13, 0x800000000LL, 0xFFFFFFFF00000000LL, a2);
      v4 = v27;
      v3 = a2;
    }
    if ( v30 != v4 )
    {
      v28 = v3;
      _libc_free((unsigned __int64)v30);
      v3 = v28;
    }
  }
  if ( *(_DWORD *)v3 != v33 || *(_BYTE *)(v3 + 4) != v34 || (v21 = *(unsigned int *)(v3 + 16), v21 != (unsigned int)v36) )
  {
LABEL_12:
    if ( v35 != v37 )
    {
      v29 = v3;
      _libc_free((unsigned __int64)v35);
      v3 = v29;
    }
    v14 = *(_QWORD *)(a1 + 16);
    v15 = v14 + 224LL * *(unsigned int *)(a1 + 24);
    if ( v14 != v15 )
    {
      while ( 1 )
      {
        if ( *(_DWORD *)v14 == *(_DWORD *)v3 && *(_BYTE *)(v14 + 4) == *(_BYTE *)(v3 + 4) )
        {
          v16 = *(unsigned int *)(v14 + 16);
          if ( v16 == *(_DWORD *)(v3 + 16) )
          {
            v17 = *(_QWORD *)(v14 + 8);
            v18 = *(_QWORD *)(v3 + 8);
            v19 = v17 + 16 * v16;
            if ( v17 == v19 )
              return sub_BA8CB0(*(_QWORD *)a1, *(_QWORD *)(v14 + 184), *(_QWORD *)(v14 + 192));
            while ( *(_DWORD *)v17 == *(_DWORD *)v18
                 && *(_DWORD *)(v17 + 4) == *(_DWORD *)(v18 + 4)
                 && *(_DWORD *)(v17 + 8) == *(_DWORD *)(v18 + 8)
                 && *(_BYTE *)(v17 + 12) == *(_BYTE *)(v18 + 12) )
            {
              v17 += 16;
              v18 += 16;
              if ( v19 == v17 )
                return sub_BA8CB0(*(_QWORD *)a1, *(_QWORD *)(v14 + 184), *(_QWORD *)(v14 + 192));
            }
          }
        }
        v14 += 224;
        if ( v15 == v14 )
          return 0;
      }
    }
    return 0;
  }
  v22 = *(_QWORD *)(v3 + 8);
  v23 = v22 + 16 * v21;
  if ( v22 != v23 )
  {
    v24 = (unsigned __int64)v35;
    while ( *(_DWORD *)v22 == *(_DWORD *)v24
         && *(_DWORD *)(v22 + 4) == *(_DWORD *)(v24 + 4)
         && *(_DWORD *)(v22 + 8) == *(_DWORD *)(v24 + 8)
         && *(_BYTE *)(v22 + 12) == *(_BYTE *)(v24 + 12) )
    {
      v22 += 16;
      v24 += 16LL;
      if ( v23 == v22 )
        goto LABEL_36;
    }
    goto LABEL_12;
  }
LABEL_36:
  if ( v35 != v37 )
    _libc_free((unsigned __int64)v35);
  v25 = *(_QWORD *)(a1 + 8);
  result = *(_BYTE **)(v25 - 32);
  if ( !result || *result || *((_QWORD *)result + 3) != *(_QWORD *)(v25 + 80) )
    return 0;
  return result;
}
