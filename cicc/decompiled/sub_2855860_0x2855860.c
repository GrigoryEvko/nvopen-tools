// Function: sub_2855860
// Address: 0x2855860
//
void __fastcall sub_2855860(__int64 a1, unsigned int a2, __int64 a3)
{
  __int64 v4; // r13
  __int64 *v5; // rdx
  __int64 v6; // rcx
  __int64 v7; // r8
  __int64 v8; // r9
  unsigned int v9; // eax
  __int64 v10; // rdx
  __int64 v11; // rbx
  __int64 v12; // r12
  __int64 v13; // rsi
  __int64 *v14; // rax
  __int64 *v15; // r14
  __int64 *v16; // r12
  char v17; // di
  __int64 v18; // rsi
  __int64 *v19; // rax
  __int64 *v20; // rax
  __int64 *v21; // rdx
  __int64 *v22; // rbx
  __int64 v23; // r14
  __int64 *v24; // r15
  char v25; // cl
  __int64 v26; // r12
  __int64 v27; // r13
  _QWORD *v28; // rax
  _QWORD *v29; // rdx
  __int64 *v30; // rax
  unsigned int v31; // ecx
  __int64 v32; // rsi
  unsigned int v33; // edx
  __int64 *v34; // rax
  __int64 v35; // r10
  unsigned __int64 v36; // rsi
  int v37; // eax
  int v38; // edi
  __int64 v39; // rax
  __int64 v41; // [rsp+8h] [rbp-88h]
  __int64 v43; // [rsp+18h] [rbp-78h]
  __int64 v44; // [rsp+18h] [rbp-78h]
  char v45[8]; // [rsp+20h] [rbp-70h] BYREF
  __int64 *v46; // [rsp+28h] [rbp-68h]
  int v47; // [rsp+30h] [rbp-60h]
  int v48; // [rsp+34h] [rbp-5Ch]
  char v49; // [rsp+3Ch] [rbp-54h]
  _BYTE v50[80]; // [rsp+40h] [rbp-50h] BYREF

  v4 = a1 + 2120;
  sub_C8CF70((__int64)v45, v50, 4, a1 + 2152, a1 + 2120);
  ++*(_QWORD *)(a1 + 2120);
  if ( *(_BYTE *)(a1 + 2148) )
    goto LABEL_6;
  v9 = 4 * (*(_DWORD *)(a1 + 2140) - *(_DWORD *)(a1 + 2144));
  v10 = *(unsigned int *)(a1 + 2136);
  if ( v9 < 0x20 )
    v9 = 32;
  if ( (unsigned int)v10 <= v9 )
  {
    memset(*(void **)(a1 + 2128), -1, 8 * v10);
LABEL_6:
    *(_QWORD *)(a1 + 2140) = 0;
    goto LABEL_7;
  }
  sub_C8C990(v4, (__int64)v50);
LABEL_7:
  v11 = *(_QWORD *)(a1 + 760);
  v12 = 112LL * *(unsigned int *)(a1 + 768);
  v43 = v11 + v12;
  if ( v11 != v11 + v12 )
  {
    while ( 1 )
    {
      v13 = *(_QWORD *)(v11 + 88);
      if ( !v13 )
        goto LABEL_14;
      if ( !*(_BYTE *)(a1 + 2148) )
        break;
      v14 = *(__int64 **)(a1 + 2128);
      v6 = *(unsigned int *)(a1 + 2140);
      v5 = &v14[v6];
      if ( v14 == v5 )
      {
LABEL_35:
        if ( (unsigned int)v6 >= *(_DWORD *)(a1 + 2136) )
          break;
        v6 = (unsigned int)(v6 + 1);
        *(_DWORD *)(a1 + 2140) = v6;
        *v5 = v13;
        ++*(_QWORD *)(a1 + 2120);
      }
      else
      {
        while ( v13 != *v14 )
        {
          if ( v5 == ++v14 )
            goto LABEL_35;
        }
      }
LABEL_14:
      v15 = *(__int64 **)(v11 + 40);
      v16 = &v15[*(unsigned int *)(v11 + 48)];
      if ( v15 != v16 )
      {
        v17 = *(_BYTE *)(a1 + 2148);
        while ( 1 )
        {
          while ( 1 )
          {
            v18 = *v15;
            if ( v17 )
              break;
LABEL_29:
            ++v15;
            sub_C8CC70(v4, v18, (__int64)v5, v6, v7, v8);
            v17 = *(_BYTE *)(a1 + 2148);
            if ( v16 == v15 )
              goto LABEL_22;
          }
          v19 = *(__int64 **)(a1 + 2128);
          v6 = *(unsigned int *)(a1 + 2140);
          v5 = &v19[v6];
          if ( v19 == v5 )
          {
LABEL_31:
            if ( (unsigned int)v6 >= *(_DWORD *)(a1 + 2136) )
              goto LABEL_29;
            v6 = (unsigned int)(v6 + 1);
            ++v15;
            *(_DWORD *)(a1 + 2140) = v6;
            *v5 = v18;
            v17 = *(_BYTE *)(a1 + 2148);
            ++*(_QWORD *)(a1 + 2120);
            if ( v16 == v15 )
              break;
          }
          else
          {
            while ( v18 != *v19 )
            {
              if ( v5 == ++v19 )
                goto LABEL_31;
            }
            if ( v16 == ++v15 )
              break;
          }
        }
      }
LABEL_22:
      v11 += 112;
      if ( v43 == v11 )
        goto LABEL_23;
    }
    sub_C8CC70(v4, v13, (__int64)v5, v6, v7, v8);
    goto LABEL_14;
  }
LABEL_23:
  v20 = v46;
  if ( !v49 )
  {
    v21 = &v46[v47];
    if ( v46 == v21 )
      goto LABEL_47;
LABEL_25:
    while ( 1 )
    {
      v22 = v20;
      if ( (unsigned __int64)*v20 < 0xFFFFFFFFFFFFFFFELL )
        break;
      if ( v21 == ++v20 )
        goto LABEL_27;
    }
    if ( v21 == v20 )
    {
LABEL_27:
      if ( v49 )
        return;
    }
    else
    {
      v23 = a1;
      v24 = v21;
      v25 = a2;
      v41 = 8LL * (a2 >> 6);
      v44 = ~(1LL << v25);
      v26 = v4;
      v27 = *v20;
      if ( !*(_BYTE *)(v23 + 2148) )
        goto LABEL_50;
LABEL_39:
      v28 = *(_QWORD **)(v23 + 2128);
      v29 = &v28[*(unsigned int *)(v23 + 2140)];
      if ( v28 == v29 )
      {
LABEL_51:
        v31 = *(_DWORD *)(a3 + 24);
        v32 = *(_QWORD *)(a3 + 8);
        if ( v31 )
        {
          v33 = (v31 - 1) & (((unsigned int)v27 >> 9) ^ ((unsigned int)v27 >> 4));
          v34 = (__int64 *)(v32 + 16LL * v33);
          v35 = *v34;
          if ( *v34 == v27 )
            goto LABEL_53;
          v37 = 1;
          while ( v35 != -4096 )
          {
            v38 = v37 + 1;
            v39 = (v31 - 1) & (v33 + v37);
            v33 = v39;
            v34 = (__int64 *)(v32 + 16 * v39);
            v35 = *v34;
            if ( *v34 == v27 )
              goto LABEL_53;
            v37 = v38;
          }
        }
        v34 = (__int64 *)(v32 + 16LL * v31);
LABEL_53:
        v36 = v34[1];
        if ( (v36 & 1) != 0 )
          v34[1] = 2
                 * (((unsigned __int64)v34[1] >> 58 << 57)
                  | v44 & (v36 >> 1) & ~(-1LL << ((unsigned __int64)v34[1] >> 58)))
                 + 1;
        else
          *(_QWORD *)(*(_QWORD *)v36 + v41) &= v44;
        goto LABEL_43;
      }
      while ( *v28 != v27 )
      {
        if ( v29 == ++v28 )
          goto LABEL_51;
      }
LABEL_43:
      while ( 1 )
      {
        v30 = v22 + 1;
        if ( v22 + 1 == v24 )
          break;
        while ( 1 )
        {
          v27 = *v30;
          v22 = v30;
          if ( (unsigned __int64)*v30 < 0xFFFFFFFFFFFFFFFELL )
            break;
          if ( v24 == ++v30 )
            goto LABEL_46;
        }
        if ( v24 == v30 )
          break;
        if ( *(_BYTE *)(v23 + 2148) )
          goto LABEL_39;
LABEL_50:
        if ( !sub_C8CA60(v26, v27) )
          goto LABEL_51;
      }
LABEL_46:
      if ( v49 )
        return;
    }
LABEL_47:
    _libc_free((unsigned __int64)v46);
    return;
  }
  v21 = &v46[v48];
  if ( v46 != v21 )
    goto LABEL_25;
}
