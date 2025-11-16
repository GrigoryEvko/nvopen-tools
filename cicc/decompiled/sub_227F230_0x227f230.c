// Function: sub_227F230
// Address: 0x227f230
//
__int64 __fastcall sub_227F230(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 result; // rax
  __int64 v5; // rdx
  __int64 v6; // rcx
  __int64 v7; // rax
  __int64 v8; // r8
  unsigned __int64 v9; // rax
  int v10; // edi
  unsigned int v11; // eax
  __int64 v12; // rsi
  void *v13; // r9
  unsigned int v14; // eax
  __int64 *v15; // r12
  _QWORD *v16; // r12
  _QWORD *i; // rbx
  __int64 v18; // rdx
  unsigned int v19; // eax
  char v20; // si
  int v21; // esi
  __int64 v22; // rax
  __int64 v23; // rdx
  __int64 v24; // rax
  unsigned __int64 v25; // rdx
  __int64 v26; // r13
  __int64 v27; // rbx
  void **v28; // rdx
  void **v29; // rax
  void **v30; // rax
  __int64 *v31; // rax
  __int64 v32; // rbx
  __int64 v33; // rax
  unsigned int v34; // eax
  __int64 v35; // [rsp+8h] [rbp-C8h]
  __int64 v37; // [rsp+18h] [rbp-B8h]
  __int64 v38; // [rsp+20h] [rbp-B0h]
  _QWORD *v39; // [rsp+28h] [rbp-A8h]
  _QWORD *v40; // [rsp+30h] [rbp-A0h]
  _QWORD *v41; // [rsp+38h] [rbp-98h]
  __int64 v42; // [rsp+40h] [rbp-90h] BYREF
  void **v43; // [rsp+48h] [rbp-88h]
  __int64 v44; // [rsp+50h] [rbp-80h]
  int v45; // [rsp+58h] [rbp-78h]
  char v46; // [rsp+5Ch] [rbp-74h]
  void *v47; // [rsp+60h] [rbp-70h] BYREF
  __int64 v48; // [rsp+70h] [rbp-60h] BYREF
  void **v49; // [rsp+78h] [rbp-58h]
  __int64 v50; // [rsp+80h] [rbp-50h]
  int v51; // [rsp+88h] [rbp-48h]
  char v52; // [rsp+8Ch] [rbp-44h]
  char v53; // [rsp+90h] [rbp-40h] BYREF

  *(_QWORD *)(sub_227ED20(a3, &qword_4FDADA8, (__int64 *)a1, a2) + 8) = a4;
  result = *(_QWORD *)(a1 + 8);
  v35 = result + 8LL * *(unsigned int *)(a1 + 16);
  if ( result != v35 )
  {
    v38 = *(_QWORD *)(a1 + 8);
    do
    {
      v5 = *(unsigned int *)(a4 + 88);
      v6 = *(_QWORD *)(a4 + 72);
      v7 = *(_QWORD *)(*(_QWORD *)v38 + 8LL);
      v37 = v7;
      if ( !(_DWORD)v5 )
        goto LABEL_20;
      v8 = (unsigned int)(v5 - 1);
      v9 = 0xBF58476D1CE4E5B9LL
         * (((unsigned __int64)(((unsigned int)&unk_4FDADB0 >> 9) ^ ((unsigned int)&unk_4FDADB0 >> 4)) << 32)
          | ((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
      v10 = 1;
      v11 = v8 & ((v9 >> 31) ^ v9);
      while ( 1 )
      {
        v12 = v6 + 24LL * v11;
        v13 = *(void **)v12;
        if ( *(_UNKNOWN **)v12 == &unk_4FDADB0 && v37 == *(_QWORD *)(v12 + 8) )
          break;
        if ( v13 == (void *)-4096LL )
        {
          if ( *(_QWORD *)(v12 + 8) == -4096 )
            goto LABEL_20;
          v34 = v10 + v11;
          ++v10;
          v11 = v8 & v34;
        }
        else
        {
          v14 = v10 + v11;
          ++v10;
          v11 = v8 & v14;
        }
      }
      if ( v12 == v6 + 24 * v5 )
        goto LABEL_20;
      v18 = *(_QWORD *)(*(_QWORD *)(v12 + 16) + 24LL);
      if ( !v18 )
        goto LABEL_20;
      v48 = 0;
      v43 = &v47;
      v49 = (void **)&v53;
      v44 = 0x100000002LL;
      v50 = 2;
      v51 = 0;
      v52 = 1;
      v45 = 0;
      v46 = 1;
      v47 = &unk_4F82400;
      v19 = *(_DWORD *)(v18 + 24);
      v42 = 1;
      v20 = *(_BYTE *)(v18 + 24);
      if ( v19 >> 1 )
      {
        v21 = v20 & 1;
        if ( v21 )
        {
          i = (_QWORD *)(v18 + 32);
          v16 = (_QWORD *)(v18 + 64);
        }
        else
        {
          v22 = *(unsigned int *)(v18 + 40);
          v6 = *(_QWORD *)(v18 + 32);
          i = (_QWORD *)v6;
          v16 = (_QWORD *)(v6 + 16 * v22);
          if ( v16 == (_QWORD *)v6 )
            goto LABEL_33;
        }
        do
        {
          if ( *i != -8192 && *i != -4096 )
            break;
          i += 2;
        }
        while ( i != v16 );
      }
      else
      {
        v21 = v20 & 1;
        if ( v21 )
        {
          v32 = v18 + 32;
          v33 = 32;
        }
        else
        {
          v32 = *(_QWORD *)(v18 + 32);
          v33 = 16LL * *(unsigned int *)(v18 + 40);
        }
        i = (_QWORD *)(v33 + v32);
        v16 = i;
      }
      if ( (_BYTE)v21 )
      {
        v23 = v18 + 32;
        v24 = 32;
        goto LABEL_34;
      }
      v6 = *(_QWORD *)(v18 + 32);
      v22 = *(unsigned int *)(v18 + 40);
LABEL_33:
      v23 = v6;
      v24 = 16 * v22;
LABEL_34:
      v39 = (_QWORD *)(v23 + v24);
      while ( v39 != i )
      {
        v25 = i[1] & 0xFFFFFFFFFFFFFFF8LL;
        if ( (i[1] & 4) != 0 )
        {
          v6 = *(_QWORD *)v25;
          v26 = *(_QWORD *)v25 + 8LL * *(unsigned int *)(v25 + 8);
          v41 = i + 2;
        }
        else
        {
          v6 = (__int64)(i + 1);
          v41 = i + 2;
          if ( !v25 )
            goto LABEL_11;
          v26 = (__int64)(i + 2);
        }
        if ( v26 != v6 )
        {
          v40 = v16;
          v15 = (__int64 *)v6;
          while ( 1 )
          {
            v27 = *v15;
            if ( v46 )
            {
              v28 = &v43[HIDWORD(v44)];
              v29 = v43;
              if ( v43 != v28 )
              {
                while ( (void *)v27 != *v29 )
                {
                  if ( v28 == ++v29 )
                    goto LABEL_46;
                }
                --HIDWORD(v44);
                v28 = (void **)v43[HIDWORD(v44)];
                *v29 = v28;
                ++v42;
              }
            }
            else
            {
              v31 = sub_C8CA60((__int64)&v42, v27);
              if ( v31 )
              {
                *v31 = -2;
                ++v45;
                ++v42;
              }
            }
LABEL_46:
            if ( !v52 )
              goto LABEL_8;
            v30 = v49;
            v28 = &v49[HIDWORD(v50)];
            if ( v49 != v28 )
            {
              while ( (void *)v27 != *v30 )
              {
                if ( v28 == ++v30 )
                  goto LABEL_52;
              }
              goto LABEL_9;
            }
LABEL_52:
            if ( HIDWORD(v50) < (unsigned int)v50 )
            {
              ++v15;
              ++HIDWORD(v50);
              *v28 = (void *)v27;
              ++v48;
              if ( (__int64 *)v26 == v15 )
              {
LABEL_10:
                v16 = v40;
                break;
              }
            }
            else
            {
LABEL_8:
              sub_C8CC70((__int64)&v48, v27, (__int64)v28, v6, v8, (__int64)v13);
LABEL_9:
              if ( (__int64 *)v26 == ++v15 )
                goto LABEL_10;
            }
          }
        }
LABEL_11:
        for ( i = v41; v16 != i; i += 2 )
        {
          if ( *i != -8192 && *i != -4096 )
            break;
        }
      }
      sub_BBE020(a4, v37, (__int64)&v42, v6);
      if ( !v52 )
        _libc_free((unsigned __int64)v49);
      if ( !v46 )
        _libc_free((unsigned __int64)v43);
LABEL_20:
      v38 += 8;
      result = v38;
    }
    while ( v35 != v38 );
  }
  return result;
}
