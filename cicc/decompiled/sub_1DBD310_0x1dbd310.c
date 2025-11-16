// Function: sub_1DBD310
// Address: 0x1dbd310
//
__int64 __fastcall sub_1DBD310(__int64 a1, __int64 a2)
{
  __int64 v4; // r13
  __int64 result; // rax
  __int64 v6; // rdx
  __int64 *v7; // rdi
  unsigned __int64 v8; // rcx
  unsigned int v9; // esi
  unsigned __int64 v10; // rsi
  __int64 v11; // r8
  unsigned __int64 v12; // rax
  __int64 v13; // rsi
  _BYTE *v14; // r8
  _BYTE *v15; // rdx
  char v16; // cl
  _BYTE *v17; // r8
  _BYTE *v18; // rcx
  _QWORD *v19; // r8
  unsigned int v20; // r9d
  __int64 v21; // r9
  __int64 v22; // rax
  unsigned __int64 v23; // r9
  __int64 v24; // r8
  unsigned __int64 v25; // rcx
  unsigned int v26; // r9d
  __int64 v27; // r14
  unsigned __int64 v28; // r15
  unsigned int v29; // r10d
  __int64 v30; // rbx
  unsigned int v31; // ecx
  __int64 v32; // rax
  __int64 *v33; // rbx
  __int64 v34; // rax
  __int64 v35; // rdx
  __int64 *v36; // rsi
  __int64 v37; // rax
  __int64 v38; // rdx
  __int64 v39; // rdx
  __int64 v40; // rax
  __int64 v41; // [rsp+8h] [rbp-38h]
  __int64 v42; // [rsp+8h] [rbp-38h]
  __int64 v43; // [rsp+8h] [rbp-38h]
  __int64 v44; // [rsp+8h] [rbp-38h]

  v4 = *(_QWORD *)a2 + 24LL * *(unsigned int *)(a2 + 8);
  result = sub_1DB3C70((__int64 *)a2, *(_QWORD *)(a1 + 24) & 0xFFFFFFFFFFFFFFF8LL);
  if ( result == v4 )
    return result;
  v6 = *(_QWORD *)result;
  v7 = (__int64 *)result;
  v8 = *(_QWORD *)(a1 + 24) & 0xFFFFFFFFFFFFFFF8LL;
  result = *(_QWORD *)result & 0xFFFFFFFFFFFFFFF8LL;
  v9 = *(_DWORD *)(v8 + 24);
  if ( *(_DWORD *)(result + 24) > v9 )
    return result;
  if ( *(_DWORD *)(result + 24) >= v9 )
    goto LABEL_32;
  result = v7[1] & 0xFFFFFFFFFFFFFFF8LL;
  v10 = *(_QWORD *)(a1 + 32) & 0xFFFFFFFFFFFFFFF8LL;
  if ( *(_DWORD *)(v10 + 24) <= *(_DWORD *)(result + 24) )
    return result;
  if ( result )
  {
    v11 = *(_QWORD *)(result + 16);
    if ( v11 )
    {
      v12 = *(_QWORD *)(result + 16);
      if ( (*(_BYTE *)(v11 + 46) & 4) != 0 )
      {
        do
          v12 = *(_QWORD *)v12 & 0xFFFFFFFFFFFFFFF8LL;
        while ( (*(_BYTE *)(v12 + 46) & 4) != 0 );
      }
      v13 = *(_QWORD *)(v11 + 24) + 24LL;
      while ( 1 )
      {
        v14 = *(_BYTE **)(v12 + 32);
        v15 = &v14[40 * *(unsigned int *)(v12 + 40)];
        if ( v14 != v15 )
          break;
        v12 = *(_QWORD *)(v12 + 8);
        if ( v13 == v12 || (*(_BYTE *)(v12 + 46) & 4) == 0 )
          goto LABEL_22;
      }
      do
      {
        while ( 1 )
        {
          if ( !*v14 )
          {
            v16 = v14[3];
            if ( (v16 & 0x10) == 0 )
              v14[3] = v16 & 0xBF;
          }
          v17 = v14 + 40;
          v18 = v15;
          if ( v17 == v15 )
            break;
          v15 = v17;
LABEL_44:
          v14 = v15;
          v15 = v18;
        }
        while ( 1 )
        {
          v12 = *(_QWORD *)(v12 + 8);
          if ( v13 == v12 || (*(_BYTE *)(v12 + 46) & 4) == 0 )
            break;
          v15 = *(_BYTE **)(v12 + 32);
          v18 = &v15[40 * *(unsigned int *)(v12 + 40)];
          if ( v15 != v18 )
            goto LABEL_44;
        }
        v14 = v15;
        v15 = v18;
LABEL_22:
        ;
      }
      while ( v14 != v15 );
      v8 = *(_QWORD *)(a1 + 24) & 0xFFFFFFFFFFFFFFF8LL;
      v10 = *(_QWORD *)(a1 + 32) & 0xFFFFFFFFFFFFFFF8LL;
    }
  }
  v19 = v7 + 3;
  if ( (__int64 *)v4 != v7 + 3 )
  {
    v6 = v7[3];
    if ( (v6 & 0xFFFFFFFFFFFFFFF8LL) == v8
      || (v20 = *(_DWORD *)(v10 + 24), *(_DWORD *)((v6 & 0xFFFFFFFFFFFFFFF8LL) + 24) >= v20) )
    {
      v21 = v7[1];
      v22 = (v21 >> 1) & 3;
      v23 = v21 & 0xFFFFFFFFFFFFFFF8LL;
      result = v10 | (2LL * (v22 != 1) + 2);
      v7[1] = result;
      if ( (_QWORD *)v4 == v19 )
        return result;
      if ( v23 != v8 )
        return result;
      result = *(_QWORD *)(a1 + 24) & 0xFFFFFFFFFFFFFFF8LL;
      if ( result != (v6 & 0xFFFFFFFFFFFFFFF8LL) )
        return result;
      v7 += 3;
LABEL_32:
      v24 = v7[2];
      v25 = *(_QWORD *)(a1 + 32) & 0xFFFFFFFFFFFFFFF8LL;
      result = v7[1] & 0xFFFFFFFFFFFFFFF8LL;
      v26 = *(_DWORD *)(result + 24);
      v27 = v25 | (2LL * (((v6 >> 1) & 3) != 1) + 2);
      v28 = v25 | (2LL * (((v6 >> 1) & 3) != 1) + 2) & 0xFFFFFFFFFFFFFFF8LL;
      v29 = *(_DWORD *)(v28 + 0x18);
      if ( v29 < v26 )
      {
        *(_QWORD *)(v24 + 8) = v27;
        *v7 = v27;
      }
      else
      {
        v41 = (v7[1] >> 1) & 3;
        v30 = 24LL * *(unsigned int *)(a2 + 8);
        v31 = *(_DWORD *)(v25 + 24) | 2;
        v32 = *(_QWORD *)(*(_QWORD *)a2 + v30 - 16);
        v33 = (__int64 *)(*(_QWORD *)a2 + v30);
        result = *(_DWORD *)((v32 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v32 >> 1) & 3;
        if ( v31 < (unsigned int)result )
        {
          v33 = v7;
          for ( result = v26 | (unsigned int)v41;
                v31 >= (unsigned int)result;
                result = *(_DWORD *)((v34 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v34 >> 1) & 3 )
          {
            v34 = v33[4];
            v33 += 3;
          }
        }
        if ( v41 == 3 || v29 <= v26 )
        {
          if ( (__int64 *)v4 != v33 && (result = *v33 & 0xFFFFFFFFFFFFFFF8LL, v28 == result) )
          {
            return sub_1DB4670(a2, v24);
          }
          else
          {
            if ( v33 != v7 + 3 )
            {
              v42 = v7[2];
              result = (__int64)memmove(v7, v7 + 3, (char *)v33 - (char *)(v7 + 3));
              v24 = v42;
            }
            *(_QWORD *)(v24 + 8) = v27;
            *(v33 - 3) = v27;
            *(v33 - 2) = v28 | 6;
            *(v33 - 1) = v24;
          }
        }
        else
        {
          if ( v7 == *(__int64 **)a2
            || *(_DWORD *)((*(v7 - 2) & 0xFFFFFFFFFFFFFFF8LL) + 24) < *(_DWORD *)((v6 & 0xFFFFFFFFFFFFFFF8LL) + 24) )
          {
            result = v7[1];
            v35 = v7[5];
            v7[3] = result;
            *(_QWORD *)(v35 + 8) = result;
          }
          else
          {
            result = v7[1];
            *(v7 - 2) = result;
          }
          v36 = v7 + 3;
          if ( (__int64 *)v4 == v33 )
          {
            if ( v33 != v36 )
            {
              v44 = v24;
              result = (__int64)memmove(v7, v36, (char *)v33 - (char *)v36);
              v24 = v44;
            }
            *(v33 - 3) = v27;
            *(v33 - 2) = v28 | 6;
            *(v33 - 1) = v24;
            *(_QWORD *)(v24 + 8) = v27;
            *(v33 - 5) = v27;
          }
          else
          {
            if ( v33 + 3 != v36 )
            {
              v43 = v24;
              memmove(v7, v7 + 3, (char *)v33 - (char *)v7);
              v24 = v43;
            }
            if ( *(_DWORD *)((*(v33 - 3) & 0xFFFFFFFFFFFFFFF8LL) + 24) >= *(_DWORD *)(v28 + 24) )
            {
              result = *v33;
              *(v33 - 3) = v27;
              *(v33 - 1) = v24;
              *(v33 - 2) = result;
              *(_QWORD *)(v24 + 8) = v27;
            }
            else
            {
              v37 = *(v33 - 1);
              v38 = *(v33 - 2);
              *v33 = v27;
              v33[2] = v37;
              v33[1] = v38;
              *(_QWORD *)(v37 + 8) = v27;
              result = *(v33 - 3);
              *(v33 - 2) = v27;
              *(v33 - 1) = v24;
              *(_QWORD *)(v24 + 8) = result;
            }
          }
        }
      }
      return result;
    }
    v39 = 24LL * *(unsigned int *)(a2 + 8);
    if ( v20 < (*(_DWORD *)((*(_QWORD *)(*(_QWORD *)a2 + v39 - 16) & 0xFFFFFFFFFFFFFFF8LL) + 24)
              | (unsigned int)(*(__int64 *)(*(_QWORD *)a2 + v39 - 16) >> 1) & 3) )
    {
      if ( v20 < (*(_DWORD *)((v7[4] & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v7[4] >> 1) & 3) )
      {
LABEL_61:
        result = v7[3];
        v7[1] = result;
        return result;
      }
      do
      {
        v40 = v19[4];
        v19 += 3;
      }
      while ( v20 >= (*(_DWORD *)((v40 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v40 >> 1) & 3) );
    }
    else
    {
      v19 = (_QWORD *)(*(_QWORD *)a2 + v39);
    }
    if ( (_QWORD *)v4 == v19 || v20 <= *(_DWORD *)((*v19 & 0xFFFFFFFFFFFFFFF8LL) + 24) )
      *(v19 - 2) = v10 | 4;
    goto LABEL_61;
  }
  result = v10 | 4;
  if ( ((v7[1] >> 1) & 3) == 1 )
    result = v10 | 2;
  v7[1] = result;
  return result;
}
