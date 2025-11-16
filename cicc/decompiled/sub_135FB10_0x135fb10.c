// Function: sub_135FB10
// Address: 0x135fb10
//
bool __fastcall sub_135FB10(__int64 a1, __int64 *a2, char a3)
{
  __int64 v4; // r14
  unsigned int v5; // r13d
  _QWORD *v6; // rcx
  __int64 i; // rax
  __int64 v9; // rsi
  __int64 v10; // rdi
  __int64 v11; // rax
  __int64 v12; // r9
  __int64 *v13; // rax
  void *v14; // rdi
  void *v15; // rsi
  __int64 v16; // rcx
  bool v17; // dl
  bool v18; // al
  char v19; // al
  __int64 *v20; // rsi
  unsigned int v21; // edi
  __int64 *v22; // rcx
  unsigned int v23; // eax
  __int64 v24; // rdx
  bool result; // al
  unsigned int v26; // edx
  unsigned int v27; // ecx
  __int64 v28; // rax
  __int64 v29; // rax
  __int64 v30; // rax
  _QWORD *v31; // r10
  __int64 v32; // r11
  void *v33; // rdi
  unsigned int v34; // eax
  __int64 v35; // rdx
  __int64 v36; // r10
  __int64 v37; // rax
  __int64 v38; // r9
  unsigned int v39; // eax
  _QWORD *v40; // [rsp+0h] [rbp-E0h]
  __int64 v41; // [rsp+8h] [rbp-D8h]
  __int64 v42; // [rsp+8h] [rbp-D8h]
  __int64 v43; // [rsp+18h] [rbp-C8h]
  bool v44; // [rsp+18h] [rbp-C8h]
  __int64 v45; // [rsp+18h] [rbp-C8h]
  bool v46; // [rsp+18h] [rbp-C8h]
  __int64 v47; // [rsp+18h] [rbp-C8h]
  __int64 v48; // [rsp+18h] [rbp-C8h]
  _QWORD *v49; // [rsp+20h] [rbp-C0h] BYREF
  __int64 v50; // [rsp+28h] [rbp-B8h]
  _QWORD v51[22]; // [rsp+30h] [rbp-B0h] BYREF

  v4 = a1 + 888;
  v5 = 8;
  v6 = v51;
  i = *a2;
  v49 = v51;
  v51[0] = i;
  v50 = 0x1000000001LL;
  LODWORD(i) = 1;
  while ( 1 )
  {
    v9 = *(_QWORD *)(a1 + 8);
    v10 = v6[(unsigned int)i - 1];
    LODWORD(v50) = i - 1;
    v11 = sub_14AD280(v10, v9, 6);
    v12 = sub_1CCAE90(v11, 1);
    v13 = *(__int64 **)(a1 + 896);
    if ( *(__int64 **)(a1 + 904) == v13 )
    {
      v20 = &v13[*(unsigned int *)(a1 + 916)];
      v21 = *(_DWORD *)(a1 + 916);
      if ( v13 != v20 )
      {
        v22 = 0;
        do
        {
          if ( v12 == *v13 )
          {
            ++*(_QWORD *)(a1 + 888);
            goto LABEL_27;
          }
          if ( *v13 == -2 )
            v22 = v13;
          ++v13;
        }
        while ( v20 != v13 );
        if ( v22 )
        {
          *v22 = v12;
          v28 = *(_QWORD *)(a1 + 888);
          --*(_DWORD *)(a1 + 920);
          v14 = *(void **)(a1 + 904);
          v16 = v28 + 1;
          v15 = *(void **)(a1 + 896);
          *(_QWORD *)(a1 + 888) = v28 + 1;
          goto LABEL_4;
        }
      }
      if ( v21 < *(_DWORD *)(a1 + 912) )
      {
        *(_DWORD *)(a1 + 916) = v21 + 1;
        *v20 = v12;
        v15 = *(void **)(a1 + 896);
        v14 = *(void **)(a1 + 904);
        v16 = *(_QWORD *)(a1 + 888) + 1LL;
        *(_QWORD *)(a1 + 888) = v16;
        goto LABEL_4;
      }
    }
    v43 = v12;
    sub_16CCBA0(v4, v12);
    v14 = *(void **)(a1 + 904);
    v15 = *(void **)(a1 + 896);
    v16 = *(_QWORD *)(a1 + 888);
    v12 = v43;
    v18 = v17;
    if ( !v17 )
    {
      *(_QWORD *)(a1 + 888) = v16 + 1;
      if ( v15 == v14 )
        goto LABEL_27;
LABEL_32:
      v26 = 4 * (*(_DWORD *)(a1 + 916) - *(_DWORD *)(a1 + 920));
      v27 = *(_DWORD *)(a1 + 912);
      if ( v26 < 0x20 )
        v26 = 32;
      if ( v27 > v26 )
      {
        v46 = v18;
        sub_16CC920(v4);
        result = v46;
        goto LABEL_28;
      }
      v24 = v27;
LABEL_26:
      memset(v14, -1, 8 * v24);
      goto LABEL_27;
    }
LABEL_4:
    v19 = *(_BYTE *)(v12 + 16);
    if ( a3 && v19 == 53 )
      goto LABEL_9;
    switch ( v19 )
    {
      case 3:
        if ( *(_DWORD *)(*(_QWORD *)v12 + 8LL) >> 8 != 4 )
        {
          v18 = *(_BYTE *)(v12 + 80) & 1;
          if ( !v18 )
          {
            *(_QWORD *)(a1 + 888) = v16 + 1;
            if ( v14 != v15 )
              goto LABEL_32;
LABEL_27:
            *(_QWORD *)(a1 + 916) = 0;
            result = 0;
            goto LABEL_28;
          }
        }
LABEL_9:
        LODWORD(i) = v50;
        break;
      case 79:
        v36 = *(_QWORD *)(v12 - 48);
        v37 = (unsigned int)v50;
        if ( (unsigned int)v50 >= HIDWORD(v50) )
        {
          v42 = *(_QWORD *)(v12 - 48);
          v48 = v12;
          sub_16CD150(&v49, v51, 0, 8);
          v37 = (unsigned int)v50;
          v36 = v42;
          v12 = v48;
        }
        v49[v37] = v36;
        i = (unsigned int)(v50 + 1);
        LODWORD(v50) = i;
        v38 = *(_QWORD *)(v12 - 24);
        if ( HIDWORD(v50) <= (unsigned int)i )
        {
          v47 = v38;
          sub_16CD150(&v49, v51, 0, 8);
          i = (unsigned int)v50;
          v38 = v47;
        }
        v49[i] = v38;
        LODWORD(i) = v50 + 1;
        LODWORD(v50) = v50 + 1;
        break;
      case 77:
        v29 = *(_DWORD *)(v12 + 20) & 0xFFFFFFF;
        if ( (unsigned int)v29 > v5 )
        {
          *(_QWORD *)(a1 + 888) = v16 + 1;
          if ( v14 == v15 )
            goto LABEL_27;
          v39 = 4 * (*(_DWORD *)(a1 + 916) - *(_DWORD *)(a1 + 920));
          v24 = *(unsigned int *)(a1 + 912);
          if ( v39 < 0x20 )
            v39 = 32;
          if ( (unsigned int)v24 <= v39 )
            goto LABEL_26;
          goto LABEL_64;
        }
        v30 = 3 * v29;
        if ( (*(_BYTE *)(v12 + 23) & 0x40) != 0 )
        {
          v31 = *(_QWORD **)(v12 - 8);
          v12 = (__int64)&v31[v30];
        }
        else
        {
          v31 = (_QWORD *)(v12 - v30 * 8);
        }
        for ( i = (unsigned int)v50; (_QWORD *)v12 != v31; LODWORD(v50) = v50 + 1 )
        {
          v32 = *v31;
          if ( (unsigned int)i >= HIDWORD(v50) )
          {
            v40 = v31;
            v41 = *v31;
            v45 = v12;
            sub_16CD150(&v49, v51, 0, 8);
            i = (unsigned int)v50;
            v31 = v40;
            v32 = v41;
            v12 = v45;
          }
          v31 += 3;
          v49[i] = v32;
          i = (unsigned int)(v50 + 1);
        }
        break;
      default:
        *(_QWORD *)(a1 + 888) = v16 + 1;
        if ( v14 == v15 )
          goto LABEL_27;
        v23 = 4 * (*(_DWORD *)(a1 + 916) - *(_DWORD *)(a1 + 920));
        v24 = *(unsigned int *)(a1 + 912);
        if ( v23 < 0x20 )
          v23 = 32;
        if ( v23 >= (unsigned int)v24 )
          goto LABEL_26;
LABEL_64:
        sub_16CC920(v4);
        result = 0;
        goto LABEL_28;
    }
    if ( !(_DWORD)i )
      break;
    if ( !--v5 )
      break;
    v6 = v49;
  }
  ++*(_QWORD *)(a1 + 888);
  v33 = *(void **)(a1 + 904);
  if ( v33 == *(void **)(a1 + 896) )
    goto LABEL_53;
  v34 = 4 * (*(_DWORD *)(a1 + 916) - *(_DWORD *)(a1 + 920));
  v35 = *(unsigned int *)(a1 + 912);
  if ( v34 < 0x20 )
    v34 = 32;
  if ( v34 < (unsigned int)v35 )
  {
    sub_16CC920(v4);
    LODWORD(i) = v50;
  }
  else
  {
    memset(v33, -1, 8 * v35);
    LODWORD(i) = v50;
LABEL_53:
    *(_QWORD *)(a1 + 916) = 0;
  }
  result = (_DWORD)i == 0;
LABEL_28:
  if ( v49 != v51 )
  {
    v44 = result;
    _libc_free((unsigned __int64)v49);
    return v44;
  }
  return result;
}
