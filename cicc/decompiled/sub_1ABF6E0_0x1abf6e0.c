// Function: sub_1ABF6E0
// Address: 0x1abf6e0
//
void __fastcall sub_1ABF6E0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned __int64 v6; // r14
  unsigned int v7; // eax
  int v8; // r9d
  unsigned int *v9; // rdi
  __int64 v10; // r12
  unsigned int v11; // ebx
  __int64 v12; // r9
  unsigned int v13; // edx
  unsigned int v14; // r8d
  __int64 *v15; // rax
  __int64 v16; // rdi
  unsigned __int64 v17; // rdx
  __int64 v18; // rax
  unsigned int v19; // esi
  __int64 v20; // r12
  int v21; // esi
  int v22; // esi
  __int64 v23; // r8
  unsigned int v24; // edx
  int v25; // eax
  __int64 *v26; // rcx
  __int64 v27; // rdi
  int v28; // edx
  int v29; // r11d
  int v30; // eax
  int v31; // esi
  int v32; // esi
  __int64 v33; // r8
  __int64 *v34; // r10
  int v35; // r11d
  unsigned int v36; // edx
  __int64 v37; // rdi
  __int64 v38; // r14
  _BYTE *v39; // r13
  __int64 v40; // rax
  int v41; // r11d
  unsigned int v42; // [rsp+8h] [rbp-F8h]
  unsigned __int64 v43; // [rsp+8h] [rbp-F8h]
  int v45; // [rsp+28h] [rbp-D8h]
  __int64 v46; // [rsp+28h] [rbp-D8h]
  __int64 v47; // [rsp+38h] [rbp-C8h] BYREF
  unsigned int *v48; // [rsp+40h] [rbp-C0h] BYREF
  __int64 v49; // [rsp+48h] [rbp-B8h]
  _BYTE s[32]; // [rsp+50h] [rbp-B0h] BYREF
  _BYTE *v51; // [rsp+70h] [rbp-90h] BYREF
  __int64 v52; // [rsp+78h] [rbp-88h]
  _BYTE v53[64]; // [rsp+80h] [rbp-80h] BYREF
  __int64 v54; // [rsp+C0h] [rbp-40h]
  char v55; // [rsp+C8h] [rbp-38h]

  v6 = sub_157EBA0(a2);
  v7 = sub_15F4D60(v6);
  v9 = (unsigned int *)s;
  v10 = v7;
  v48 = (unsigned int *)s;
  v49 = 0x800000000LL;
  if ( v7 > 8 )
  {
    sub_16CD150((__int64)&v48, s, v7, 4, (int)&v48, v8);
    v9 = v48;
  }
  LODWORD(v49) = v10;
  if ( 4 * v10 )
    memset(v9, 0, 4 * v10);
  v55 = 0;
  v11 = 0;
  v51 = v53;
  v52 = 0x400000000LL;
  v54 = 0;
  v45 = sub_15F4D60(v6);
  if ( v45 )
  {
    while ( 1 )
    {
      LODWORD(v47) = v11;
      v18 = sub_15F4DF0(v6, v11);
      v19 = *(_DWORD *)(a3 + 24);
      v20 = v18;
      if ( !v19 )
        break;
      v12 = *(_QWORD *)(a3 + 8);
      v13 = ((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4);
      v14 = (v19 - 1) & v13;
      v15 = (__int64 *)(v12 + 16LL * v14);
      v16 = *v15;
      if ( v20 != *v15 )
      {
        v29 = 1;
        v26 = 0;
        while ( v16 != -8 )
        {
          if ( !v26 && v16 == -16 )
            v26 = v15;
          v14 = (v19 - 1) & (v29 + v14);
          v15 = (__int64 *)(v12 + 16LL * v14);
          v16 = *v15;
          if ( v20 == *v15 )
            goto LABEL_8;
          ++v29;
        }
        if ( !v26 )
          v26 = v15;
        v30 = *(_DWORD *)(a3 + 16);
        ++*(_QWORD *)a3;
        v25 = v30 + 1;
        if ( 4 * v25 < 3 * v19 )
        {
          if ( v19 - *(_DWORD *)(a3 + 20) - v25 <= v19 >> 3 )
          {
            v42 = v13;
            sub_1956860(a3, v19);
            v31 = *(_DWORD *)(a3 + 24);
            if ( !v31 )
            {
LABEL_59:
              ++*(_DWORD *)(a3 + 16);
              BUG();
            }
            v32 = v31 - 1;
            v33 = *(_QWORD *)(a3 + 8);
            v34 = 0;
            v35 = 1;
            v36 = v32 & v42;
            v25 = *(_DWORD *)(a3 + 16) + 1;
            v26 = (__int64 *)(v33 + 16LL * (v32 & v42));
            v37 = *v26;
            if ( v20 != *v26 )
            {
              while ( v37 != -8 )
              {
                if ( !v34 && v37 == -16 )
                  v34 = v26;
                v36 = v32 & (v35 + v36);
                v26 = (__int64 *)(v33 + 16LL * v36);
                v37 = *v26;
                if ( v20 == *v26 )
                  goto LABEL_14;
                ++v35;
              }
LABEL_33:
              if ( v34 )
                v26 = v34;
            }
          }
LABEL_14:
          *(_DWORD *)(a3 + 16) = v25;
          if ( *v26 != -8 )
            --*(_DWORD *)(a3 + 20);
          *v26 = v20;
          v26[1] = 0;
          goto LABEL_17;
        }
LABEL_12:
        sub_1956860(a3, 2 * v19);
        v21 = *(_DWORD *)(a3 + 24);
        if ( !v21 )
          goto LABEL_59;
        v22 = v21 - 1;
        v23 = *(_QWORD *)(a3 + 8);
        v24 = v22 & (((unsigned int)v20 >> 9) ^ ((unsigned int)v20 >> 4));
        v25 = *(_DWORD *)(a3 + 16) + 1;
        v26 = (__int64 *)(v23 + 16LL * v24);
        v27 = *v26;
        if ( v20 != *v26 )
        {
          v41 = 1;
          v34 = 0;
          while ( v27 != -8 )
          {
            if ( !v34 && v27 == -16 )
              v34 = v26;
            v24 = v22 & (v41 + v24);
            v26 = (__int64 *)(v23 + 16LL * v24);
            v27 = *v26;
            if ( v20 == *v26 )
              goto LABEL_14;
            ++v41;
          }
          goto LABEL_33;
        }
        goto LABEL_14;
      }
LABEL_8:
      v17 = v15[1];
      if ( v17 )
      {
        ++v11;
        sub_1370BE0((__int64)&v51, (unsigned int *)&v47, v17, 1u);
        if ( v45 == v11 )
          goto LABEL_18;
      }
      else
      {
LABEL_17:
        v28 = v11++;
        sub_1379150(a4, a2, v28, 0);
        if ( v45 == v11 )
          goto LABEL_18;
      }
    }
    ++*(_QWORD *)a3;
    goto LABEL_12;
  }
LABEL_18:
  if ( v54 )
  {
    sub_1372DF0((__int64)&v51);
    if ( (_DWORD)v52 )
    {
      v43 = v6;
      v46 = 16LL * (unsigned int)v52;
      v38 = 0;
      do
      {
        v39 = &v51[v38];
        v38 += 16;
        v48[*((unsigned int *)v39 + 1)] = *((_QWORD *)v39 + 1);
        sub_16AF710(&v47, *((_DWORD *)v39 + 2), v54);
        sub_1379150(a4, a2, *((_DWORD *)v39 + 1), v47);
      }
      while ( v38 != v46 );
      v6 = v43;
    }
    v47 = sub_16498A0(v6);
    v40 = sub_161BD30(&v47, v48, (unsigned int)v49);
    sub_1625C10(v6, 2, v40);
  }
  if ( v51 != v53 )
    _libc_free((unsigned __int64)v51);
  if ( v48 != (unsigned int *)s )
    _libc_free((unsigned __int64)v48);
}
