// Function: sub_2B25A00
// Address: 0x2b25a00
//
_QWORD *__fastcall sub_2B25A00(_QWORD *a1, char *a2, unsigned __int64 *a3)
{
  unsigned __int64 v6; // rax
  __int64 v7; // rcx
  bool v8; // dl
  unsigned int v9; // esi
  int v10; // edx
  unsigned __int64 v11; // rax
  __int64 v13; // rax
  unsigned __int64 v14; // r15
  unsigned __int8 *v15; // rax
  unsigned __int64 v16; // rax
  unsigned __int64 v17; // rcx
  char v18; // al
  unsigned __int64 v19; // rax
  bool v20; // al
  unsigned __int64 v21; // rbx
  char v22; // al
  size_t v23; // rdx
  __int64 v24; // rax
  __int64 v25; // r9
  __int64 v26; // r13
  void *v27; // rax
  unsigned int v28; // r14d
  unsigned __int64 v29; // r13
  size_t v30; // rdx
  size_t v31; // rdx
  char *v32; // r8
  char *v33; // rdi
  unsigned __int64 v34; // rsi
  __int64 v35; // rdx
  __int64 v36; // rcx
  __int64 v37; // r8
  __int64 v38; // r9
  unsigned __int64 *v39; // rax
  unsigned int v40; // esi
  unsigned __int64 v41; // rdi
  unsigned __int64 v42; // rdx
  unsigned __int64 v43; // rax
  __int64 v44; // [rsp+8h] [rbp-58h]
  char *v45; // [rsp+8h] [rbp-58h]
  char *v46; // [rsp+8h] [rbp-58h]
  unsigned __int64 v47; // [rsp+18h] [rbp-48h] BYREF
  unsigned __int64 *v48; // [rsp+20h] [rbp-40h] BYREF
  unsigned __int64 *v49[7]; // [rsp+28h] [rbp-38h] BYREF

  v6 = *a3;
  v7 = *a3 & 1;
  if ( (*a3 & 1) != 0 )
    v8 = v6 >> 58 == 0;
  else
    v8 = *(_DWORD *)(v6 + 64) == 0;
  v9 = 1;
  if ( !v8 )
  {
    if ( (_BYTE)v7 )
      v9 = v6 >> 58;
    else
      v9 = *(_DWORD *)(v6 + 64);
  }
  sub_B48880((__int64 *)&v47, v9, 1u);
  v10 = (unsigned __int8)*a2;
  if ( (unsigned int)(v10 - 12) <= 1 )
    goto LABEL_7;
  v13 = *((_QWORD *)a2 + 1);
  if ( *(_BYTE *)(v13 + 8) == 17 )
  {
    if ( (unsigned __int8)v10 > 0x15u )
    {
      v19 = *a3;
      if ( (*a3 & 1) != 0 )
        v20 = v19 >> 58 == 0;
      else
        v20 = *(_DWORD *)(v19 + 64) == 0;
      if ( !v20 )
      {
        if ( (_BYTE)v10 == 91 )
        {
          v32 = a2;
          do
          {
            v33 = v32;
            v32 = (char *)*((_QWORD *)v32 - 12);
            if ( (unsigned int)**((unsigned __int8 **)v33 - 8) - 12 > 1 )
            {
              v46 = v32;
              v39 = (unsigned __int64 *)sub_2B18C70(v33, 0);
              v32 = v46;
              v49[0] = v39;
              v40 = (unsigned int)v39;
              if ( !BYTE4(v39) )
              {
                sub_2B254D0(&v47);
                *a1 = v47;
                return a1;
              }
              v41 = *a3;
              v42 = (unsigned int)v39;
              if ( (*a3 & 1) != 0 )
                v43 = v41 >> 58;
              else
                v43 = *(unsigned int *)(v41 + 64);
              if ( v42 < v43 && !(unsigned __int8)sub_2B0D930(v41, v40) )
                sub_2B0DA70((__int64 *)&v47, v40);
            }
          }
          while ( *v32 == 91 );
          if ( v32 == a2 )
            goto LABEL_30;
          v34 = *a3;
          if ( (*a3 & 1) != 0 )
            v34 >>= 58;
          else
            LODWORD(v34) = *(_DWORD *)(v34 + 64);
          v45 = v32;
          sub_B48880((__int64 *)&v48, v34, 0);
          sub_2B25A00(v49, v45, &v48);
          sub_2B25100(&v47, (unsigned __int64 *)v49, v35, v36, v37, v38);
          sub_228BF40(v49);
          sub_228BF40(&v48);
          v11 = v47;
        }
        else
        {
LABEL_30:
          sub_2B254D0(&v47);
          v11 = v47;
        }
        goto LABEL_8;
      }
      v11 = v47;
      if ( (v47 & 1) != 0 )
      {
        v11 = v47 & 0xFC00000000000000LL | 1;
        goto LABEL_8;
      }
      v31 = 8LL * *(unsigned int *)(v47 + 8);
      if ( !v31 )
        goto LABEL_8;
      memset(*(void **)v47, 0, v31);
LABEL_7:
      v11 = v47;
LABEL_8:
      *a1 = v11;
      return a1;
    }
    v14 = 0;
    v44 = *(unsigned int *)(v13 + 32);
    if ( !(_DWORD)v44 )
      goto LABEL_7;
    while ( 1 )
    {
      v15 = (unsigned __int8 *)sub_AD69F0((unsigned __int8 *)a2, (unsigned int)v14);
      if ( !v15 || (unsigned int)*v15 - 12 <= 1 )
        goto LABEL_16;
      v16 = *a3;
      if ( (*a3 & 1) != 0 )
        break;
      if ( !*(_DWORD *)(v16 + 64) )
      {
LABEL_24:
        if ( (v47 & 1) != 0 )
          v47 = 2 * ((v47 >> 58 << 57) | ~(1LL << v14) & ~(-1LL << (v47 >> 58)) & (v47 >> 1)) + 1;
        else
          *(_QWORD *)(*(_QWORD *)v47 + 8LL * ((unsigned int)v14 >> 6)) &= ~(1LL << v14);
        goto LABEL_16;
      }
      if ( *(unsigned int *)(v16 + 64) > v14 )
      {
        v18 = (*(_QWORD *)(*(_QWORD *)v16 + 8LL * ((unsigned int)v14 >> 6)) >> v14) & 1;
LABEL_23:
        if ( !v18 )
          goto LABEL_24;
      }
LABEL_16:
      if ( ++v14 == v44 )
        goto LABEL_7;
    }
    v17 = v16 >> 58;
    if ( !(v16 >> 58) )
      goto LABEL_24;
    if ( v17 <= v14 )
      goto LABEL_16;
    v18 = ((~(-1LL << v17) & (v16 >> 1)) >> v14) & 1;
    goto LABEL_23;
  }
  v21 = v47;
  if ( (v47 & 1) != 0 )
  {
    v21 = v47 & 0xFC00000000000000LL | 1;
    v47 = v21;
    v22 = v21 & 1;
    goto LABEL_36;
  }
  v23 = 8LL * *(unsigned int *)(v47 + 8);
  if ( v23 )
  {
    memset(*(void **)v47, 0, v23);
    v21 = v47;
    v22 = v47 & 1;
LABEL_36:
    *a1 = 1;
    if ( v22 )
    {
      *a1 = v21;
      return a1;
    }
    goto LABEL_42;
  }
  *a1 = 1;
LABEL_42:
  v24 = sub_22077B0(0x48u);
  v26 = v24;
  if ( v24 )
  {
    v27 = (void *)(v24 + 16);
    *(_QWORD *)v26 = v27;
    *(_QWORD *)(v26 + 8) = 0x600000000LL;
    v28 = *(_DWORD *)(v21 + 8);
    if ( v28 && v26 != v21 )
    {
      v30 = 8LL * v28;
      if ( v28 <= 6
        || (sub_C8D5F0(v26, v27, v28, 8u, v28, v25), v27 = *(void **)v26, (v30 = 8LL * *(unsigned int *)(v21 + 8)) != 0) )
      {
        memcpy(v27, *(const void **)v21, v30);
      }
      *(_DWORD *)(v26 + 8) = v28;
    }
    *(_DWORD *)(v26 + 64) = *(_DWORD *)(v21 + 64);
  }
  *a1 = v26;
  v29 = v47;
  if ( (v47 & 1) == 0 && v47 )
  {
    if ( *(_QWORD *)v47 != v47 + 16 )
      _libc_free(*(_QWORD *)v47);
    j_j___libc_free_0(v29);
  }
  return a1;
}
