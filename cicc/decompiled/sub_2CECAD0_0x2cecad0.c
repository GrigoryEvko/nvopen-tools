// Function: sub_2CECAD0
// Address: 0x2cecad0
//
__int64 __fastcall sub_2CECAD0(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v8; // rsi
  __int64 v9; // rdx
  int v10; // r8d
  unsigned int v11; // r13d
  __int64 v12; // rdi
  __int64 *v13; // rax
  __int64 v14; // rcx
  __int64 v15; // rdx
  unsigned int v16; // eax
  __int64 result; // rax
  char v18; // cl
  _QWORD *v19; // rbx
  _QWORD *v20; // rdx
  _QWORD *v21; // r13
  _QWORD *v22; // rdi
  _QWORD *v23; // rax
  __int64 v24; // rsi
  __int64 v25; // rcx
  __int64 v26; // r14
  unsigned int v27; // edx
  __int64 v28; // rax
  __int64 v29; // rsi
  _QWORD *v30; // rax
  _QWORD *v31; // rdx
  char v32; // di
  __int64 v33; // r9
  unsigned int v34; // r11d
  int i; // r10d
  __int64 *v36; // r10
  unsigned __int64 *v37; // r11
  int v38; // r10d
  unsigned __int64 *v39; // r9
  int v40; // eax
  int v41; // edx
  unsigned __int64 v42; // rdi
  int v43; // eax
  int v44; // eax
  __int64 v45; // rsi
  unsigned int v46; // r13d
  unsigned __int64 v47; // rcx
  int v48; // r8d
  unsigned __int64 *v49; // rdi
  int v50; // eax
  int v51; // eax
  __int64 v52; // rsi
  int v53; // r8d
  unsigned int v54; // r13d
  unsigned __int64 v55; // rcx
  int v56; // [rsp+Ch] [rbp-34h]

  v8 = *(unsigned int *)(a3 + 24);
  v9 = *(_QWORD *)(a3 + 8);
  if ( !(_DWORD)v8 )
    goto LABEL_4;
  v10 = v8 - 1;
  v11 = ((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4);
  LODWORD(v12) = (v8 - 1) & v11;
  v13 = (__int64 *)(v9 + 16LL * (unsigned int)v12);
  v14 = *v13;
  if ( *v13 != a2 )
  {
    v33 = *v13;
    v34 = (v8 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    for ( i = 1; ; i = v56 )
    {
      if ( v33 == -4096 )
        goto LABEL_4;
      v34 = v10 & (i + v34);
      v56 = i + 1;
      v36 = (__int64 *)(v9 + 16LL * v34);
      v33 = *v36;
      if ( *v36 == a2 )
        break;
    }
    if ( v36 == (__int64 *)(v9 + 16LL * (unsigned int)v8) )
      goto LABEL_4;
    v37 = (unsigned __int64 *)(v9 + 16LL * (v10 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4))));
    v38 = 1;
    v39 = 0;
    while ( v14 != -4096 )
    {
      if ( v14 != -8192 || v39 )
        v37 = v39;
      v12 = v10 & (unsigned int)(v12 + v38);
      v13 = (__int64 *)(v9 + 16 * v12);
      v14 = *v13;
      if ( *v13 == a2 )
        return *((unsigned int *)v13 + 2);
      ++v38;
      v39 = v37;
      v37 = (unsigned __int64 *)(v9 + 16 * v12);
    }
    if ( !v39 )
      v39 = (unsigned __int64 *)v13;
    v40 = *(_DWORD *)(a3 + 16);
    ++*(_QWORD *)a3;
    v41 = v40 + 1;
    if ( 4 * (v40 + 1) >= (unsigned int)(3 * v8) )
    {
      sub_D39D40(a3, 2 * v8);
      v43 = *(_DWORD *)(a3 + 24);
      if ( v43 )
      {
        v44 = v43 - 1;
        v45 = *(_QWORD *)(a3 + 8);
        v46 = v44 & v11;
        v41 = *(_DWORD *)(a3 + 16) + 1;
        v39 = (unsigned __int64 *)(v45 + 16LL * v46);
        v47 = *v39;
        if ( *v39 == a2 )
          goto LABEL_56;
        v48 = 1;
        v49 = 0;
        while ( v47 != -4096 )
        {
          if ( !v49 && v47 == -8192 )
            v49 = v39;
          v46 = v44 & (v48 + v46);
          v39 = (unsigned __int64 *)(v45 + 16LL * v46);
          v47 = *v39;
          if ( *v39 == a2 )
            goto LABEL_56;
          ++v48;
        }
LABEL_73:
        if ( v49 )
          v39 = v49;
        goto LABEL_56;
      }
    }
    else
    {
      if ( (int)v8 - *(_DWORD *)(a3 + 20) - v41 > (unsigned int)v8 >> 3 )
      {
LABEL_56:
        *(_DWORD *)(a3 + 16) = v41;
        if ( *v39 != -4096 )
          --*(_DWORD *)(a3 + 20);
        *v39 = a2;
        *((_DWORD *)v39 + 2) = 0;
        return 0;
      }
      sub_D39D40(a3, v8);
      v50 = *(_DWORD *)(a3 + 24);
      if ( v50 )
      {
        v51 = v50 - 1;
        v52 = *(_QWORD *)(a3 + 8);
        v53 = 1;
        v54 = v51 & v11;
        v41 = *(_DWORD *)(a3 + 16) + 1;
        v49 = 0;
        v39 = (unsigned __int64 *)(v52 + 16LL * v54);
        v55 = *v39;
        if ( *v39 == a2 )
          goto LABEL_56;
        while ( v55 != -4096 )
        {
          if ( !v49 && v55 == -8192 )
            v49 = v39;
          v54 = v51 & (v53 + v54);
          v39 = (unsigned __int64 *)(v52 + 16LL * v54);
          v55 = *v39;
          if ( *v39 == a2 )
            goto LABEL_56;
          ++v53;
        }
        goto LABEL_73;
      }
    }
    ++*(_DWORD *)(a3 + 16);
    BUG();
  }
  if ( v13 != (__int64 *)(v9 + 16 * v8) )
    return *((unsigned int *)v13 + 2);
LABEL_4:
  v15 = *(_QWORD *)(a2 + 8);
  if ( *(_BYTE *)(v15 + 8) == 14 )
  {
    v16 = *(_DWORD *)(v15 + 8) >> 8;
    if ( v16 )
    {
      switch ( v16 )
      {
        case 1u:
          return 1;
        case 2u:
          return 15;
        case 3u:
          goto LABEL_62;
        case 4u:
          goto LABEL_63;
        case 5u:
          return 8;
        case 6u:
          goto LABEL_61;
        default:
          goto LABEL_33;
      }
    }
  }
  v18 = *(_BYTE *)a2;
  if ( *(_BYTE *)a2 != 22 )
  {
    if ( v18 == 3 )
    {
      v27 = *(_DWORD *)(v15 + 8);
      v16 = v27 >> 8;
      if ( v27 <= 0x6FF )
      {
        if ( v16 )
        {
          switch ( v16 )
          {
            case 1u:
              return 1;
            case 3u:
              goto LABEL_62;
            case 4u:
              goto LABEL_63;
            case 5u:
              return 8;
            case 6u:
              goto LABEL_61;
            default:
              return 15;
          }
        }
        return 15;
      }
    }
    else
    {
      result = 15;
      if ( v18 != 5 )
        return result;
      v16 = sub_2CDD760(a2, a4);
      if ( v16 <= 6 )
      {
        if ( v16 )
        {
          switch ( v16 )
          {
            case 1u:
              return 1;
            case 3u:
LABEL_62:
              result = 2;
              break;
            case 4u:
LABEL_63:
              result = 4;
              break;
            case 5u:
              return 8;
            case 6u:
LABEL_61:
              result = 32;
              break;
            default:
              return 15;
          }
          return result;
        }
        return 15;
      }
    }
LABEL_33:
    if ( v16 == 101 )
      return 16;
    return 15;
  }
  if ( unk_50142AD && (unsigned __int8)sub_CE9220(a4) && !(unsigned __int8)sub_B2D680(a2) )
    return 1;
  if ( (unsigned __int8)sub_B2D680(a2) && !(unsigned __int8)sub_CE9220(a4) )
    return 8;
  v19 = *(_QWORD **)(a1 + 8);
  if ( !v19 )
    return 15;
  v20 = (_QWORD *)v19[2];
  v21 = v19 + 1;
  if ( !v20 )
    return 15;
  v22 = v19 + 1;
  v23 = (_QWORD *)v19[2];
  do
  {
    while ( 1 )
    {
      v24 = v23[2];
      v25 = v23[3];
      if ( v23[4] >= a2 )
        break;
      v23 = (_QWORD *)v23[3];
      if ( !v25 )
        goto LABEL_19;
    }
    v22 = v23;
    v23 = (_QWORD *)v23[2];
  }
  while ( v24 );
LABEL_19:
  result = 15;
  if ( v21 != v22 )
  {
    v26 = (__int64)(v19 + 1);
    if ( v22[4] <= a2 )
    {
      do
      {
        if ( v20[4] < a2 )
        {
          v20 = (_QWORD *)v20[3];
        }
        else
        {
          v26 = (__int64)v20;
          v20 = (_QWORD *)v20[2];
        }
      }
      while ( v20 );
      if ( v21 == (_QWORD *)v26 || *(_QWORD *)(v26 + 32) > a2 )
      {
        v28 = sub_22077B0(0x30u);
        v29 = v26;
        *(_QWORD *)(v28 + 32) = a2;
        v26 = v28;
        *(_DWORD *)(v28 + 40) = 0;
        v30 = sub_2CBBA50(v19, v29, (unsigned __int64 *)(v28 + 32));
        if ( v31 )
        {
          v32 = v21 == v31 || v30 || v31[4] > a2;
          sub_220F040(v32, v26, v31, v19 + 1);
          ++v19[5];
        }
        else
        {
          v42 = v26;
          v26 = (__int64)v30;
          j_j___libc_free_0(v42);
        }
      }
      result = *(unsigned int *)(v26 + 40);
      if ( (unsigned int)result > 6 )
        return (unsigned int)((_DWORD)result == 101) + 15;
      if ( (_DWORD)result )
      {
        switch ( (int)result )
        {
          case 1:
          case 4:
            return result;
          case 3:
            goto LABEL_62;
          case 5:
            return 8;
          case 6:
            goto LABEL_61;
          default:
            return 15;
        }
        return result;
      }
      return 15;
    }
  }
  return result;
}
