// Function: sub_15AE530
// Address: 0x15ae530
//
void __fastcall sub_15AE530(__int64 a1, __int64 a2)
{
  int v3; // eax
  int v4; // edx
  __int64 v5; // rdi
  unsigned int v6; // eax
  __int64 v7; // rcx
  __int64 v8; // rdi
  __int64 v9; // rdx
  unsigned int v10; // esi
  _QWORD *v11; // rax
  int v12; // r11d
  __int64 *v13; // r10
  unsigned int v14; // ecx
  __int64 *v15; // rdx
  __int64 v16; // rax
  __int64 v17; // r13
  unsigned int v18; // edx
  __int64 v19; // rsi
  int v20; // eax
  __int64 v21; // rax
  _BYTE **v22; // rbx
  unsigned int v23; // ecx
  _BYTE *v24; // r9
  _BYTE *v25; // r12
  int v26; // ecx
  int v27; // ecx
  __int64 v28; // r10
  unsigned int v29; // esi
  _BYTE *v30; // r9
  int v31; // r11d
  _BYTE *v32; // rcx
  int v33; // r9d
  __int64 *v34; // rdi
  unsigned int v35; // ebx
  __int64 v36; // rcx
  int v37; // r10d
  int v38; // r9d
  int v39; // r11d
  __int64 *v40; // r9
  __int64 v41; // [rsp-E8h] [rbp-E8h] BYREF
  __int64 v42; // [rsp-E0h] [rbp-E0h]
  __int64 v43; // [rsp-D8h] [rbp-D8h]
  __int64 i; // [rsp-D0h] [rbp-D0h]
  _QWORD *v45; // [rsp-C8h] [rbp-C8h] BYREF
  __int64 v46; // [rsp-C0h] [rbp-C0h]
  _QWORD v47[23]; // [rsp-B8h] [rbp-B8h] BYREF

  if ( a2 )
  {
    v3 = *(_DWORD *)(a1 + 24);
    if ( v3 )
    {
      v4 = v3 - 1;
      v5 = *(_QWORD *)(a1 + 8);
      v6 = (v3 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v7 = *(_QWORD *)(v5 + 16LL * v6);
      if ( v7 == a2 )
        return;
      v38 = 1;
      while ( v7 != -4 )
      {
        v6 = v4 & (v38 + v6);
        v7 = *(_QWORD *)(v5 + 16LL * v6);
        if ( v7 == a2 )
          return;
        ++v38;
      }
    }
    v8 = 0;
    v47[0] = a2;
    v9 = 1;
    v10 = 0;
    v46 = 0x1000000001LL;
    v11 = v47;
    v45 = v47;
    v41 = 0;
    v42 = 0;
    v43 = 0;
    for ( i = 0; ; v10 = i )
    {
      v17 = v11[v9 - 1];
      if ( v10 )
      {
        v12 = 1;
        v13 = 0;
        v14 = (v10 - 1) & (((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4));
        v15 = (__int64 *)(v8 + 8LL * v14);
        v16 = *v15;
        if ( v17 == *v15 )
        {
LABEL_7:
          sub_15AD010(a1, v17);
          v8 = v42;
          v9 = (unsigned int)(v46 - 1);
          LODWORD(v46) = v46 - 1;
          goto LABEL_8;
        }
        while ( v16 != -8 )
        {
          if ( v16 != -16 || v13 )
            v15 = v13;
          v14 = (v10 - 1) & (v12 + v14);
          v16 = *(_QWORD *)(v8 + 8LL * v14);
          if ( v17 == v16 )
            goto LABEL_7;
          ++v12;
          v13 = v15;
          v15 = (__int64 *)(v8 + 8LL * v14);
        }
        if ( !v13 )
          v13 = v15;
        ++v41;
        v20 = v43 + 1;
        if ( 4 * ((int)v43 + 1) < 3 * v10 )
        {
          if ( v10 - (v20 + HIDWORD(v43)) <= v10 >> 3 )
          {
            sub_15AE380((__int64)&v41, v10);
            if ( !(_DWORD)i )
            {
LABEL_79:
              LODWORD(v43) = v43 + 1;
              BUG();
            }
            v33 = 1;
            v34 = 0;
            v35 = (i - 1) & (((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4));
            v13 = (__int64 *)(v42 + 8LL * v35);
            v36 = *v13;
            v20 = v43 + 1;
            if ( v17 != *v13 )
            {
              while ( v36 != -8 )
              {
                if ( !v34 && v36 == -16 )
                  v34 = v13;
                v35 = (i - 1) & (v33 + v35);
                v13 = (__int64 *)(v42 + 8LL * v35);
                v36 = *v13;
                if ( v17 == *v13 )
                  goto LABEL_14;
                ++v33;
              }
              if ( v34 )
                v13 = v34;
            }
          }
          goto LABEL_14;
        }
      }
      else
      {
        ++v41;
      }
      sub_15AE380((__int64)&v41, 2 * v10);
      if ( !(_DWORD)i )
        goto LABEL_79;
      v18 = (i - 1) & (((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4));
      v13 = (__int64 *)(v42 + 8LL * v18);
      v19 = *v13;
      v20 = v43 + 1;
      if ( v17 != *v13 )
      {
        v39 = 1;
        v40 = 0;
        while ( v19 != -8 )
        {
          if ( !v40 && v19 == -16 )
            v40 = v13;
          v18 = (i - 1) & (v39 + v18);
          v13 = (__int64 *)(v42 + 8LL * v18);
          v19 = *v13;
          if ( v17 == *v13 )
            goto LABEL_14;
          ++v39;
        }
        if ( v40 )
          v13 = v40;
      }
LABEL_14:
      LODWORD(v43) = v20;
      if ( *v13 != -8 )
        --HIDWORD(v43);
      *v13 = v17;
      v21 = 8LL * *(unsigned int *)(v17 + 8);
      v22 = (_BYTE **)(v17 - v21);
      if ( v17 != v17 - v21 )
      {
        v8 = v42;
        v9 = (unsigned int)v46;
        while ( 1 )
        {
          v25 = *v22;
          if ( !*v22 || (unsigned __int8)(*v25 - 4) > 0x1Eu )
            goto LABEL_19;
          if ( (_DWORD)i )
          {
            v23 = (i - 1) & (((unsigned int)v25 >> 9) ^ ((unsigned int)v25 >> 4));
            v24 = *(_BYTE **)(v8 + 8LL * v23);
            if ( v25 == v24 )
              goto LABEL_19;
            v37 = 1;
            while ( v24 != (_BYTE *)-8LL )
            {
              v23 = (i - 1) & (v37 + v23);
              v24 = *(_BYTE **)(v8 + 8LL * v23);
              if ( v25 == v24 )
                goto LABEL_19;
              ++v37;
            }
          }
          v26 = *(_DWORD *)(a1 + 24);
          if ( !v26 )
            goto LABEL_27;
          v27 = v26 - 1;
          v28 = *(_QWORD *)(a1 + 8);
          v29 = v27 & (((unsigned int)v25 >> 9) ^ ((unsigned int)v25 >> 4));
          v30 = *(_BYTE **)(v28 + 16LL * v29);
          if ( v30 != v25 )
          {
            v31 = 1;
            while ( v30 != (_BYTE *)-4LL )
            {
              v29 = v27 & (v31 + v29);
              v30 = *(_BYTE **)(v28 + 16LL * v29);
              if ( v25 == v30 )
                goto LABEL_19;
              ++v31;
            }
LABEL_27:
            if ( (*(_BYTE *)v17 != 17
               || (v32 = *(_BYTE **)(v17 + 8 * (7LL - *(unsigned int *)(v17 + 8)))) == 0
               || v25 != v32)
              && *v25 != 16 )
            {
              if ( (unsigned int)v9 >= HIDWORD(v46) )
              {
                sub_16CD150(&v45, v47, 0, 8);
                v9 = (unsigned int)v46;
              }
              v45[v9] = v25;
              v8 = v42;
              v9 = (unsigned int)(v46 + 1);
              LODWORD(v46) = v46 + 1;
            }
          }
LABEL_19:
          if ( (_BYTE **)v17 == ++v22 )
            goto LABEL_8;
        }
      }
      v9 = (unsigned int)v46;
      v8 = v42;
LABEL_8:
      if ( !(_DWORD)v9 )
      {
        j___libc_free_0(v8);
        if ( v45 != v47 )
          _libc_free((unsigned __int64)v45);
        return;
      }
      v11 = v45;
    }
  }
}
