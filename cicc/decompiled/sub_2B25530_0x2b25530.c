// Function: sub_2B25530
// Address: 0x2b25530
//
_QWORD *__fastcall sub_2B25530(_QWORD *a1, __int64 a2, unsigned __int64 *a3)
{
  unsigned __int64 v6; // rax
  __int64 v7; // rcx
  bool v8; // dl
  unsigned int v9; // esi
  unsigned __int8 v10; // dl
  __int64 v11; // rax
  unsigned __int64 v12; // r15
  unsigned int v13; // r14d
  _BYTE *v14; // rax
  unsigned __int64 v15; // rax
  unsigned __int64 v16; // rcx
  unsigned __int64 v17; // rax
  unsigned __int64 v19; // rbx
  size_t v20; // rdx
  char v21; // al
  unsigned __int64 v22; // rax
  bool v23; // al
  size_t v24; // rdx
  __int64 v25; // rax
  __int64 v26; // r9
  __int64 v27; // r13
  void *v28; // rax
  unsigned int v29; // r14d
  unsigned __int64 v30; // r13
  size_t v31; // rdx
  char *v32; // r8
  char *v33; // rdi
  unsigned __int64 *v34; // rax
  unsigned int v35; // esi
  unsigned __int64 v36; // rdi
  unsigned __int64 v37; // rdx
  unsigned __int64 v38; // rax
  unsigned __int64 v39; // rsi
  __int64 v40; // rdx
  __int64 v41; // rcx
  __int64 v42; // r8
  __int64 v43; // r9
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
  v10 = *(_BYTE *)a2;
  if ( *(_BYTE *)a2 == 13 )
    goto LABEL_19;
  v11 = *(_QWORD *)(a2 + 8);
  if ( *(_BYTE *)(v11 + 8) == 17 )
  {
    if ( v10 > 0x15u )
    {
      v22 = *a3;
      if ( (*a3 & 1) != 0 )
        v23 = v22 >> 58 == 0;
      else
        v23 = *(_DWORD *)(v22 + 64) == 0;
      if ( !v23 && v10 == 91 )
      {
        v32 = (char *)a2;
        do
        {
          v33 = v32;
          v32 = (char *)*((_QWORD *)v32 - 12);
          if ( **((_BYTE **)v33 - 8) != 13 )
          {
            v45 = v32;
            v34 = (unsigned __int64 *)sub_2B18C70(v33, 0);
            v32 = v45;
            v49[0] = v34;
            v35 = (unsigned int)v34;
            if ( !BYTE4(v34) )
            {
              sub_2B254D0(&v47);
              *a1 = v47;
              return a1;
            }
            v36 = *a3;
            v37 = (unsigned int)v34;
            if ( (*a3 & 1) != 0 )
              v38 = v36 >> 58;
            else
              v38 = *(unsigned int *)(v36 + 64);
            if ( v37 < v38 && !(unsigned __int8)sub_2B0D930(v36, v35) )
              sub_2B0DA70((__int64 *)&v47, v35);
          }
        }
        while ( *v32 == 91 );
        if ( v32 == (char *)a2 )
          goto LABEL_40;
        v39 = *a3;
        if ( (*a3 & 1) != 0 )
          v39 >>= 58;
        else
          LODWORD(v39) = *(_DWORD *)(v39 + 64);
        v46 = v32;
        sub_B48880((__int64 *)&v48, v39, 0);
        sub_2B25530(v49, v46, &v48);
        sub_2B25100(&v47, (unsigned __int64 *)v49, v40, v41, v42, v43);
        sub_228BF40(v49);
        sub_228BF40(&v48);
        v17 = v47;
        goto LABEL_20;
      }
LABEL_40:
      v17 = v47;
      if ( (v47 & 1) != 0 )
      {
        v17 = v47 & 0xFC00000000000000LL | 1;
        goto LABEL_20;
      }
      v24 = 8LL * *(unsigned int *)(v47 + 8);
      if ( !v24 )
      {
LABEL_20:
        *a1 = v17;
        return a1;
      }
      memset(*(void **)v47, 0, v24);
    }
    else
    {
      v12 = 0;
      v44 = *(unsigned int *)(v11 + 32);
      if ( (_DWORD)v44 )
      {
        while ( 1 )
        {
          v13 = v12;
          v14 = (_BYTE *)sub_AD69F0((unsigned __int8 *)a2, (unsigned int)v12);
          if ( !v14 || *v14 == 13 )
            goto LABEL_18;
          v15 = *a3;
          if ( (*a3 & 1) != 0 )
          {
            v16 = v15 >> 58;
            if ( v15 >> 58 && (v16 <= v12 || (((~(-1LL << v16) & (v15 >> 1)) >> v12) & 1) != 0) )
              goto LABEL_18;
          }
          else if ( *(_DWORD *)(v15 + 64)
                 && (*(unsigned int *)(v15 + 64) <= v12
                  || ((*(_QWORD *)(*(_QWORD *)v15 + 8LL * ((unsigned int)v12 >> 6)) >> v12) & 1) != 0) )
          {
            goto LABEL_18;
          }
          if ( (v47 & 1) != 0 )
          {
            v47 = 2 * ((v47 >> 58 << 57) | ~(1LL << v12) & ~(-1LL << (v47 >> 58)) & (v47 >> 1)) + 1;
LABEL_18:
            if ( ++v12 == v44 )
              break;
          }
          else
          {
            ++v12;
            *(_QWORD *)(*(_QWORD *)v47 + 8LL * (v13 >> 6)) &= ~(1LL << v13);
            if ( v12 == v44 )
              break;
          }
        }
      }
    }
LABEL_19:
    v17 = v47;
    goto LABEL_20;
  }
  v19 = v47;
  if ( (v47 & 1) != 0 )
  {
    v19 = v47 & 0xFC00000000000000LL | 1;
    v47 = v19;
    v21 = v19 & 1;
LABEL_26:
    *a1 = 1;
    if ( v21 )
    {
      *a1 = v19;
      return a1;
    }
    goto LABEL_44;
  }
  v20 = 8LL * *(unsigned int *)(v47 + 8);
  if ( v20 )
  {
    memset(*(void **)v47, 0, v20);
    v19 = v47;
    v21 = v47 & 1;
    goto LABEL_26;
  }
  *a1 = 1;
LABEL_44:
  v25 = sub_22077B0(0x48u);
  v27 = v25;
  if ( v25 )
  {
    v28 = (void *)(v25 + 16);
    *(_QWORD *)v27 = v28;
    *(_QWORD *)(v27 + 8) = 0x600000000LL;
    v29 = *(_DWORD *)(v19 + 8);
    if ( v29 && v27 != v19 )
    {
      v31 = 8LL * v29;
      if ( v29 <= 6
        || (sub_C8D5F0(v27, v28, v29, 8u, v29, v26), v28 = *(void **)v27, (v31 = 8LL * *(unsigned int *)(v19 + 8)) != 0) )
      {
        memcpy(v28, *(const void **)v19, v31);
      }
      *(_DWORD *)(v27 + 8) = v29;
    }
    *(_DWORD *)(v27 + 64) = *(_DWORD *)(v19 + 64);
  }
  *a1 = v27;
  v30 = v47;
  if ( (v47 & 1) == 0 && v47 )
  {
    if ( *(_QWORD *)v47 != v47 + 16 )
      _libc_free(*(_QWORD *)v47);
    j_j___libc_free_0(v30);
  }
  return a1;
}
