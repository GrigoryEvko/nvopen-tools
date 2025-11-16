// Function: sub_1DBDCF0
// Address: 0x1dbdcf0
//
__int64 __fastcall sub_1DBDCF0(_QWORD *a1, __int64 a2, int a3, int a4)
{
  __int64 result; // rax
  __int64 v9; // rdx
  char *v10; // rbx
  unsigned __int64 v11; // rdi
  unsigned __int64 v12; // rsi
  unsigned int v13; // ecx
  char *v14; // rcx
  unsigned __int64 v15; // rsi
  __int64 v16; // r13
  char *v17; // rax
  __int64 v18; // r9
  char *v19; // r14
  unsigned __int64 v20; // rdx
  __int64 v21; // rsi
  signed __int64 v22; // rdx
  unsigned int v23; // edx
  char *v24; // r14
  unsigned __int64 v25; // rax
  __int64 v26; // rcx
  __int64 v27; // rsi
  _BYTE *v28; // rdi
  _BYTE *v29; // rdx
  char v30; // cl
  _BYTE *v31; // rdi
  _BYTE *v32; // rcx
  __int64 v33; // rsi
  __int64 v34; // r12
  __int64 v35; // rdx
  __int64 v36; // [rsp+0h] [rbp-50h]
  char *v37; // [rsp+8h] [rbp-48h]
  __int64 v38; // [rsp+10h] [rbp-40h]
  __int64 v39; // [rsp+18h] [rbp-38h]

  v39 = *(_QWORD *)a2 + 24LL * *(unsigned int *)(a2 + 8);
  result = sub_1DB3C70((__int64 *)a2, a1[3] & 0xFFFFFFFFFFFFFFF8LL);
  if ( result == v39 )
    return result;
  v9 = *(_QWORD *)result;
  v10 = (char *)result;
  v11 = a1[3] & 0xFFFFFFFFFFFFFFF8LL;
  v12 = *(_QWORD *)result & 0xFFFFFFFFFFFFFFF8LL;
  result = v11;
  v13 = *(_DWORD *)(v12 + 24);
  if ( v13 > *(_DWORD *)(v11 + 24) )
    return result;
  if ( v13 < *(_DWORD *)(v11 + 24) )
  {
    if ( v11 != (*((_QWORD *)v10 + 1) & 0xFFFFFFFFFFFFFFF8LL) )
      return result;
    v21 = v12 | 6;
    v22 = (2LL * (((*((__int64 *)v10 + 1) >> 1) & 3) != 1) + 2) | a1[4] & 0xFFFFFFFFFFFFFFF8LL;
    if ( (v13 | 3) < (*(_DWORD *)((v22 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v22 >> 1) & 3) )
      v21 = (2LL * (((*((__int64 *)v10 + 1) >> 1) & 3) != 1) + 2) | a1[4] & 0xFFFFFFFFFFFFFFF8LL;
    result = sub_1DBD810(a1, v21, a3, a4);
    *((_QWORD *)v10 + 1) = result;
    if ( (char *)v39 == v10 + 24 )
      return result;
    v9 = *((_QWORD *)v10 + 3);
    result = a1[3] & 0xFFFFFFFFFFFFFFF8LL;
    if ( result != (v9 & 0xFFFFFFFFFFFFFFF8LL) )
      return result;
    v14 = v10;
    v10 += 24;
  }
  else
  {
    v14 = v10 - 24;
    if ( v10 == *(char **)a2 )
      v14 = (char *)v39;
  }
  v37 = v14;
  v38 = *((_QWORD *)v10 + 2);
  v15 = a1[4] & 0xFFFFFFFFFFFFFFF8LL;
  v36 = (*((__int64 *)v10 + 1) >> 1) & 3;
  v16 = v15 | (2LL * (((v9 >> 1) & 3) != 1) + 2);
  v17 = (char *)sub_1DB3C70((__int64 *)a2, v15 | 4);
  v18 = v38;
  v19 = v17;
  v20 = a1[4] & 0xFFFFFFFFFFFFFFF8LL;
  result = *(_QWORD *)v17 & 0xFFFFFFFFFFFFFFF8LL;
  if ( result == v20 )
  {
    v33 = v38;
    if ( v36 != 3 )
    {
      *(_QWORD *)(v38 + 8) = v16;
      *(_QWORD *)v10 = v16;
      v33 = *((_QWORD *)v19 + 2);
    }
    return sub_1DB4670(a2, v33);
  }
  else if ( v36 == 3 )
  {
    if ( v37 == (char *)v39
      || (v23 = *(_DWORD *)(v20 + 24), *(_DWORD *)(result + 24) >= v23)
      || v23 >= *(_DWORD *)((*((_QWORD *)v19 + 1) & 0xFFFFFFFFFFFFFFF8LL) + 24) )
    {
      if ( v10 != v19 )
      {
        memmove(v19 + 24, v19, v10 - v19);
        v18 = v38;
      }
      *(_QWORD *)v19 = v16;
      *((_QWORD *)v19 + 2) = v18;
      result = v16 & 0xFFFFFFFFFFFFFFF8LL | 6;
      *((_QWORD *)v19 + 1) = result;
      *(_QWORD *)(v18 + 8) = v16;
    }
    else
    {
      if ( v10 != v19 )
      {
        memmove(v19 + 24, v19, v10 - v19);
        v18 = v38;
      }
      *((_QWORD *)v19 + 5) = v18;
      v24 = v19 + 48;
      v25 = v16 & 0xFFFFFFFFFFFFFFF8LL | 4;
      *((_QWORD *)v24 - 5) = v25;
      *((_QWORD *)v24 - 3) = v25;
      for ( *(_QWORD *)(v18 + 8) = v16; v10 >= v24; v24 += 24 )
        *((_QWORD *)v24 + 2) = v18;
      result = a1[4] & 0xFFFFFFFFFFFFFFF8LL;
      if ( result )
      {
        v26 = *(_QWORD *)(result + 16);
        if ( v26 )
        {
          result = *(_QWORD *)(result + 16);
          if ( (*(_BYTE *)(v26 + 46) & 4) != 0 )
          {
            do
              result = *(_QWORD *)result & 0xFFFFFFFFFFFFFFF8LL;
            while ( (*(_BYTE *)(result + 46) & 4) != 0 );
          }
          v27 = *(_QWORD *)(v26 + 24) + 24LL;
          while ( 1 )
          {
            v28 = *(_BYTE **)(result + 32);
            v29 = &v28[40 * *(unsigned int *)(result + 40)];
            if ( v28 != v29 )
              break;
            result = *(_QWORD *)(result + 8);
            if ( v27 == result || (*(_BYTE *)(result + 46) & 4) == 0 )
              goto LABEL_43;
          }
          do
          {
            while ( 1 )
            {
              if ( !*v28 )
              {
                v30 = v28[3];
                if ( (v30 & 0x10) != 0 )
                  v28[3] = v30 & 0xBF;
              }
              v31 = v28 + 40;
              v32 = v29;
              if ( v31 == v29 )
                break;
              v29 = v31;
LABEL_58:
              v28 = v29;
              v29 = v32;
            }
            while ( 1 )
            {
              result = *(_QWORD *)(result + 8);
              if ( v27 == result || (*(_BYTE *)(result + 46) & 4) == 0 )
                break;
              v29 = *(_BYTE **)(result + 32);
              v32 = &v29[40 * *(unsigned int *)(result + 40)];
              if ( v29 != v32 )
                goto LABEL_58;
            }
            v28 = v29;
            v29 = v32;
LABEL_43:
            ;
          }
          while ( v28 != v29 );
        }
      }
    }
  }
  else if ( v37 == (char *)v39 )
  {
    *(_QWORD *)v10 = v16;
    *(_QWORD *)(v38 + 8) = v16;
  }
  else if ( *(_DWORD *)((v16 & 0xFFFFFFFFFFFFFFF8LL) + 24) < *(_DWORD *)((*(_QWORD *)v37 & 0xFFFFFFFFFFFFFFF8LL) + 24) )
  {
    v34 = *((_QWORD *)v37 + 2);
    *(_QWORD *)(*((_QWORD *)v10 + 2) + 8LL) = *(_QWORD *)v37;
    *(_QWORD *)v10 = *(_QWORD *)v37;
    if ( v37 != v19 )
      memmove(&v10[-(v37 - v19)], v19, v37 - v19);
    v35 = *((_QWORD *)v19 + 3);
    result = *(unsigned int *)((a1[4] & 0xFFFFFFFFFFFFFFF8LL) + 24);
    if ( *(_DWORD *)((v35 & 0xFFFFFFFFFFFFFFF8LL) + 24) >= (unsigned int)result )
    {
      *(_QWORD *)v19 = v16;
      *((_QWORD *)v19 + 1) = v35;
      *((_QWORD *)v19 + 2) = v34;
    }
    else
    {
      result = *((_QWORD *)v19 + 5);
      *(_QWORD *)v19 = v35;
      *((_QWORD *)v19 + 1) = v16;
      *((_QWORD *)v19 + 2) = result;
      *((_QWORD *)v19 + 3) = v16;
      *((_QWORD *)v19 + 5) = v34;
    }
    *(_QWORD *)(v34 + 8) = v16;
  }
  else
  {
    *(_QWORD *)v10 = v16;
    *(_QWORD *)(v38 + 8) = v16;
    result = a1[4] & 0xFFFFFFFFFFFFFFF8LL;
    if ( *(_DWORD *)(result + 24) < *(_DWORD *)((*((_QWORD *)v37 + 1) & 0xFFFFFFFFFFFFFFF8LL) + 24) )
    {
      result |= 4uLL;
      *((_QWORD *)v37 + 1) = result;
    }
  }
  return result;
}
