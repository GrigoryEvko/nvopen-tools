// Function: sub_1220BF0
// Address: 0x1220bf0
//
_QWORD *__fastcall sub_1220BF0(_QWORD *a1, __int64 a2)
{
  __int64 v4; // r13
  unsigned int v5; // r14d
  _QWORD *v6; // r15
  unsigned int v7; // eax
  __int64 v8; // rdx
  __int64 v9; // rcx
  size_t v10; // rcx
  size_t v11; // r11
  size_t v12; // rdx
  unsigned int v13; // eax
  size_t v14; // r13
  size_t v15; // r14
  size_t v16; // rdx
  int v17; // eax
  bool v18; // sf
  __int64 v19; // r13
  __int64 v20; // rax
  _BYTE *v21; // rsi
  __int64 v22; // rdx
  __int64 *v23; // rdi
  _BYTE *v24; // rsi
  __int64 v25; // rdx
  unsigned int v26; // eax
  __int64 *v27; // rsi
  _QWORD *v28; // rdi
  void *v29; // r13
  __int64 v30; // rax
  __int64 v31; // rax
  __int64 v32; // rdx
  __int64 v33; // r9
  __int64 v34; // rdi
  __int64 v35; // rcx
  unsigned int v37; // eax
  size_t v38; // r13
  size_t v39; // r14
  size_t v40; // rdx
  int v41; // eax
  unsigned int v42; // edi
  __int64 v43; // r13
  __int64 v44; // rdi
  _QWORD *v45; // rdi
  __int64 v46; // rdi
  __int64 v47; // rdx
  __int64 v48; // rax
  void **v49; // r12
  __int64 v50; // [rsp+8h] [rbp-58h]
  _QWORD *v51; // [rsp+18h] [rbp-48h]
  size_t v52; // [rsp+20h] [rbp-40h]
  __int64 v53; // [rsp+20h] [rbp-40h]
  size_t v54; // [rsp+28h] [rbp-38h]
  __int64 v55; // [rsp+28h] [rbp-38h]
  __int64 v56; // [rsp+28h] [rbp-38h]
  __int64 v57; // [rsp+28h] [rbp-38h]

  v4 = a1[2];
  v51 = a1 + 1;
  if ( !v4 )
  {
    v6 = a1 + 1;
    goto LABEL_27;
  }
  v5 = *(_DWORD *)a2;
  v6 = a1 + 1;
  while ( 1 )
  {
    if ( *(_DWORD *)(v4 + 32) != v5 )
    {
      LOBYTE(v7) = *(_DWORD *)(v4 + 32) < (int)v5;
      goto LABEL_4;
    }
    if ( v5 <= 1 )
      break;
    v10 = *(_QWORD *)(v4 + 72);
    v11 = *(_QWORD *)(a2 + 40);
    v12 = v11;
    if ( v10 <= v11 )
      v12 = *(_QWORD *)(v4 + 72);
    if ( v12 )
    {
      v52 = *(_QWORD *)(a2 + 40);
      v54 = *(_QWORD *)(v4 + 72);
      v13 = memcmp(*(const void **)(v4 + 64), *(const void **)(a2 + 32), v12);
      v10 = v54;
      v11 = v52;
      if ( v13 )
      {
        v7 = v13 >> 31;
        goto LABEL_4;
      }
    }
    v35 = v10 - v11;
    if ( v35 < 0x80000000LL )
    {
      if ( v35 > (__int64)0xFFFFFFFF7FFFFFFFLL )
      {
        LOBYTE(v7) = (int)v35 < 0;
        goto LABEL_4;
      }
      v9 = *(_QWORD *)(v4 + 24);
LABEL_5:
      v8 = v9;
      goto LABEL_6;
    }
    v8 = *(_QWORD *)(v4 + 16);
    v6 = (_QWORD *)v4;
LABEL_6:
    if ( !v8 )
      goto LABEL_16;
LABEL_7:
    v4 = v8;
  }
  LOBYTE(v7) = *(_DWORD *)(v4 + 48) < *(_DWORD *)(a2 + 16);
LABEL_4:
  v8 = *(_QWORD *)(v4 + 16);
  v9 = *(_QWORD *)(v4 + 24);
  if ( (_BYTE)v7 )
    goto LABEL_5;
  v6 = (_QWORD *)v4;
  if ( v8 )
    goto LABEL_7;
LABEL_16:
  if ( v51 == v6 )
    goto LABEL_27;
  if ( v5 == *((_DWORD *)v6 + 8) )
  {
    if ( v5 <= 1 )
    {
      if ( *(_DWORD *)(a2 + 16) < *((_DWORD *)v6 + 12) )
        goto LABEL_27;
    }
    else
    {
      v14 = *(_QWORD *)(a2 + 40);
      v15 = v6[9];
      v16 = v15;
      if ( v14 <= v15 )
        v16 = *(_QWORD *)(a2 + 40);
      if ( v16 && (v17 = memcmp(*(const void **)(a2 + 32), (const void *)v6[8], v16), v18 = v17 < 0, v17) )
      {
LABEL_26:
        if ( v18 )
          goto LABEL_27;
      }
      else
      {
        v19 = v14 - v15;
        if ( v19 <= 0x7FFFFFFF )
        {
          if ( v19 >= (__int64)0xFFFFFFFF80000000LL )
          {
            v18 = (int)v19 < 0;
            goto LABEL_26;
          }
LABEL_27:
          v53 = (__int64)v6;
          v20 = sub_22077B0(200);
          v21 = *(_BYTE **)(a2 + 32);
          v6 = (_QWORD *)v20;
          v55 = v20 + 32;
          v22 = (__int64)&v21[*(_QWORD *)(a2 + 40)];
          v23 = (__int64 *)(v20 + 64);
          *(_DWORD *)(v20 + 32) = *(_DWORD *)a2;
          *(_QWORD *)(v20 + 40) = *(_QWORD *)(a2 + 8);
          *(_DWORD *)(v20 + 48) = *(_DWORD *)(a2 + 16);
          *(_QWORD *)(v20 + 56) = *(_QWORD *)(a2 + 24);
          v20 += 80;
          *v23 = v20;
          v50 = v20;
          sub_12060D0(v23, v21, v22);
          v24 = *(_BYTE **)(a2 + 64);
          v25 = *(_QWORD *)(a2 + 72);
          v6[12] = v6 + 14;
          sub_12060D0(v6 + 12, v24, (__int64)&v24[v25]);
          v26 = *(_DWORD *)(a2 + 104);
          *((_DWORD *)v6 + 34) = v26;
          if ( v26 > 0x40 )
            sub_C43780((__int64)(v6 + 16), (const void **)(a2 + 96));
          else
            v6[16] = *(_QWORD *)(a2 + 96);
          *((_BYTE *)v6 + 140) = *(_BYTE *)(a2 + 108);
          v27 = (__int64 *)(a2 + 112);
          v28 = v6 + 18;
          v29 = sub_C33340();
          if ( *(void **)(a2 + 112) == v29 )
            sub_C3C790(v28, (_QWORD **)v27);
          else
            sub_C33EB0(v28, v27);
          v30 = *(_QWORD *)(a2 + 136);
          v6[22] = 0;
          v6[24] = 0;
          v6[21] = v30;
          *((_BYTE *)v6 + 184) = *(_BYTE *)(a2 + 152);
          v31 = sub_12207E0(a1, v53, v55);
          v56 = v31;
          v33 = v32;
          if ( v32 )
          {
            if ( v51 == (_QWORD *)v32 || v31 )
            {
LABEL_34:
              v34 = 1;
              goto LABEL_35;
            }
            v37 = *((_DWORD *)v6 + 8);
            if ( v37 == *(_DWORD *)(v32 + 32) )
            {
              if ( v37 <= 1 )
              {
                v34 = *((_DWORD *)v6 + 12) < *(_DWORD *)(v32 + 48);
              }
              else
              {
                v38 = v6[9];
                v40 = *(_QWORD *)(v32 + 72);
                v39 = v40;
                if ( v38 <= v40 )
                  v40 = v6[9];
                if ( v40 )
                {
                  v57 = v33;
                  v41 = memcmp((const void *)v6[8], *(const void **)(v33 + 64), v40);
                  v33 = v57;
                  v42 = v41;
                  if ( v41 )
                    goto LABEL_53;
                }
                v43 = v38 - v39;
                v34 = 0;
                if ( v43 <= 0x7FFFFFFF )
                {
                  if ( v43 < (__int64)0xFFFFFFFF80000000LL )
                    goto LABEL_34;
                  v42 = v43;
LABEL_53:
                  v34 = v42 >> 31;
                }
              }
            }
            else
            {
              v34 = (int)v37 < *(_DWORD *)(v32 + 32);
            }
LABEL_35:
            sub_220F040(v34, v6, v33, v51);
            ++a1[5];
          }
          else
          {
            if ( v29 == (void *)v6[18] )
            {
              v47 = v6[19];
              if ( v47 )
              {
                v48 = 24LL * *(_QWORD *)(v47 - 8);
                v49 = (void **)(v47 + v48);
                if ( v47 != v47 + v48 )
                {
                  do
                  {
                    v49 -= 3;
                    if ( v29 == *v49 )
                      sub_969EE0((__int64)v49);
                    else
                      sub_C338F0((__int64)v49);
                  }
                  while ( (void **)v6[19] != v49 );
                }
                j_j_j___libc_free_0_0(v49 - 1);
              }
            }
            else
            {
              sub_C338F0((__int64)(v6 + 18));
            }
            if ( *((_DWORD *)v6 + 34) > 0x40u )
            {
              v44 = v6[16];
              if ( v44 )
                j_j___libc_free_0_0(v44);
            }
            v45 = (_QWORD *)v6[12];
            if ( v6 + 14 != v45 )
              j_j___libc_free_0(v45, v6[14] + 1LL);
            v46 = v6[8];
            if ( v50 != v46 )
              j_j___libc_free_0(v46, v6[10] + 1LL);
            j_j___libc_free_0(v6, 200);
            v6 = (_QWORD *)v56;
          }
        }
      }
    }
  }
  else if ( (signed int)v5 < *((_DWORD *)v6 + 8) )
  {
    goto LABEL_27;
  }
  return v6 + 24;
}
