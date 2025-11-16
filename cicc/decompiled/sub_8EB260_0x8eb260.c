// Function: sub_8EB260
// Address: 0x8eb260
//
unsigned __int64 __fastcall sub_8EB260(unsigned __int8 *a1, int a2, int a3, __int64 a4)
{
  int v8; // esi
  unsigned __int64 result; // rax
  int v10; // r10d
  __int64 v11; // rdx
  __int64 v12; // rcx
  unsigned __int64 v13; // rax
  unsigned __int64 v14; // rdx
  int v15; // edx
  unsigned __int64 v16; // rax
  unsigned __int64 v17; // rcx
  __int64 v18; // rsi
  __int64 v19; // rcx
  unsigned __int8 *v20; // rax
  unsigned __int8 *v21; // r15
  unsigned __int8 *v22; // r14
  __int64 v23; // rcx
  unsigned __int64 v24; // rax
  unsigned __int8 *v25; // rbx
  char *v26; // rax
  __int64 v27; // rcx
  char v28; // dl
  int v29; // r12d
  __int64 v30; // rcx
  unsigned __int64 v31; // rax
  __int64 v32; // rcx
  unsigned __int64 v33; // rax
  unsigned __int64 v34; // rdx
  __int64 v35; // rdx
  __int64 v36; // rax
  unsigned __int8 *v37; // rdi
  __int64 v38; // rcx
  unsigned __int64 v39; // rdx
  __int64 v40; // rcx
  unsigned __int64 v41; // rdx
  unsigned __int64 v42; // rax
  __int64 v43; // rax
  unsigned __int64 v44; // rdx
  unsigned __int8 *v45; // rax
  __int64 v46; // rax
  unsigned __int64 v47; // rdx
  unsigned __int8 *v48; // rax
  __int64 v49; // rcx
  unsigned __int64 v50; // rax
  unsigned __int64 v51; // rdx
  int v52; // [rsp+8h] [rbp-38h]
  int v53; // [rsp+Ch] [rbp-34h]
  int v54; // [rsp+Ch] [rbp-34h]

  while ( 2 )
  {
    v8 = 0;
    while ( 1 )
    {
      result = *a1;
      if ( (_BYTE)result == 75 )
      {
        v8 |= 1u;
        goto LABEL_5;
      }
      if ( (_BYTE)result != 86 )
        break;
      v8 |= 2u;
LABEL_5:
      ++a1;
    }
    if ( (_BYTE)result == 114 )
    {
      v8 |= 4u;
      goto LABEL_5;
    }
    v10 = v8 | a2;
    if ( (_BYTE)result == 83 )
    {
      if ( a1[1] != 116 )
        return sub_8EC8F0((_DWORD)a1, 2, v10, a3, 0, 0, 0, a4);
      return result;
    }
    if ( (unsigned __int8)(result - 67) > 0x12u )
      goto LABEL_11;
    v19 = 307201;
    if ( _bittest64(&v19, (unsigned int)(result - 67)) )
    {
      ++a1;
      if ( (_BYTE)result == 85 )
      {
        ++*(_QWORD *)(a4 + 32);
        v20 = sub_8E72C0(a1, 0, a4);
        --*(_QWORD *)(a4 + 32);
        a1 = v20;
      }
      a3 = 1;
      goto LABEL_36;
    }
    if ( (_BYTE)result == 77 )
    {
      ++*(_QWORD *)(a4 + 32);
      ++*(_QWORD *)(a4 + 48);
      v37 = a1 + 1;
      a1 = (unsigned __int8 *)sub_8E9FF0((__int64)(a1 + 1), 0, 0, 0, 1u, a4);
      sub_8EB260(v37, 0, 0, a4);
      --*(_QWORD *)(a4 + 48);
      a3 = 1;
      --*(_QWORD *)(a4 + 32);
LABEL_36:
      a2 = 0;
      continue;
    }
    break;
  }
  if ( (_BYTE)result == 70 )
    goto LABEL_39;
  if ( (_BYTE)result != 68 )
  {
LABEL_11:
    if ( (_BYTE)result != 65 )
      return result;
    v11 = *(_QWORD *)(a4 + 32);
    if ( !a3 )
    {
LABEL_13:
      if ( v11 )
        goto LABEL_19;
      goto LABEL_14;
    }
    if ( v11 )
      goto LABEL_19;
    v40 = *(_QWORD *)(a4 + 8);
    v41 = v40 + 1;
    if ( !*(_DWORD *)(a4 + 28) )
    {
      v42 = *(_QWORD *)(a4 + 16);
      if ( v42 > v41 )
      {
        *(_BYTE *)(*(_QWORD *)a4 + v40) = 41;
        v11 = *(_QWORD *)(a4 + 32);
        v43 = *(_QWORD *)(a4 + 8) + 1LL;
        goto LABEL_91;
      }
      *(_DWORD *)(a4 + 28) = 1;
      if ( v42 )
      {
        *(_BYTE *)(*(_QWORD *)a4 + v42 - 1) = 0;
        v11 = *(_QWORD *)(a4 + 32);
        v43 = *(_QWORD *)(a4 + 8) + 1LL;
LABEL_91:
        *(_QWORD *)(a4 + 8) = v43;
        goto LABEL_13;
      }
    }
    *(_QWORD *)(a4 + 8) = v41;
LABEL_14:
    v12 = *(_QWORD *)(a4 + 8);
    v13 = v12 + 1;
    if ( !*(_DWORD *)(a4 + 28) )
    {
      v14 = *(_QWORD *)(a4 + 16);
      if ( v14 > v13 )
      {
        *(_BYTE *)(*(_QWORD *)a4 + v12) = 91;
        v13 = *(_QWORD *)(a4 + 8) + 1LL;
      }
      else
      {
        *(_DWORD *)(a4 + 28) = 1;
        if ( v14 )
        {
          *(_BYTE *)(*(_QWORD *)a4 + v14 - 1) = 0;
          v13 = *(_QWORD *)(a4 + 8) + 1LL;
        }
      }
    }
    *(_QWORD *)(a4 + 8) = v13;
LABEL_19:
    v15 = a1[1];
    if ( (unsigned int)(v15 - 48) > 9 )
    {
      if ( (_BYTE)v15 == 95 )
      {
        v16 = *(_QWORD *)(a4 + 32);
        ++a1;
        goto LABEL_117;
      }
      ++*(_QWORD *)(a4 + 48);
      v48 = sub_8E74B0(a1 + 1, a4);
      --*(_QWORD *)(a4 + 48);
      a1 = v48;
      LOBYTE(v15) = *v48;
      v16 = *(_QWORD *)(a4 + 32);
    }
    else
    {
      v16 = *(_QWORD *)(a4 + 32);
      ++a1;
      do
      {
        ++a1;
        if ( !v16 )
        {
          v18 = *(_QWORD *)(a4 + 8);
          v17 = v18 + 1;
          if ( !*(_DWORD *)(a4 + 28) )
          {
            v16 = *(_QWORD *)(a4 + 16);
            if ( v16 <= v17 )
            {
              *(_DWORD *)(a4 + 28) = 1;
              if ( v16 )
              {
                *(_BYTE *)(*(_QWORD *)a4 + v16 - 1) = 0;
                v17 = *(_QWORD *)(a4 + 8) + 1LL;
                v16 = *(_QWORD *)(a4 + 32);
              }
            }
            else
            {
              *(_BYTE *)(*(_QWORD *)a4 + v18) = v15;
              v17 = *(_QWORD *)(a4 + 8) + 1LL;
              v16 = *(_QWORD *)(a4 + 32);
            }
          }
          *(_QWORD *)(a4 + 8) = v17;
        }
        v15 = *a1;
      }
      while ( (unsigned int)(v15 - 48) <= 9 );
    }
    if ( (_BYTE)v15 != 95 )
    {
      if ( !*(_DWORD *)(a4 + 24) )
      {
        ++v16;
        ++*(_QWORD *)(a4 + 48);
        *(_DWORD *)(a4 + 24) = 1;
        *(_QWORD *)(a4 + 32) = v16;
      }
      goto LABEL_108;
    }
LABEL_117:
    ++a1;
LABEL_108:
    if ( !v16 )
    {
      v49 = *(_QWORD *)(a4 + 8);
      v50 = v49 + 1;
      if ( !*(_DWORD *)(a4 + 28) )
      {
        v51 = *(_QWORD *)(a4 + 16);
        if ( v51 > v50 )
        {
          *(_BYTE *)(*(_QWORD *)a4 + v49) = 93;
          v50 = *(_QWORD *)(a4 + 8) + 1LL;
        }
        else
        {
          *(_DWORD *)(a4 + 28) = 1;
          if ( v51 )
          {
            *(_BYTE *)(*(_QWORD *)a4 + v51 - 1) = 0;
            v50 = *(_QWORD *)(a4 + 8) + 1LL;
          }
        }
      }
      *(_QWORD *)(a4 + 8) = v50;
    }
    a3 = 0;
    goto LABEL_36;
  }
  result = a1[1];
  if ( (a1[1] & 0xDF) == 0x4F || (_BYTE)result == 119 )
  {
    if ( (_BYTE)result == 79 )
    {
      ++*(_QWORD *)(a4 + 32);
      v21 = a1 + 2;
      v52 = a3;
      v54 = v10;
      v45 = sub_8E74B0(a1 + 2, a4);
      v10 = v54;
      a3 = v52;
      a1 = v45;
      v46 = *(_QWORD *)(a4 + 32);
      *(_QWORD *)(a4 + 32) = v46 - 1;
      if ( *a1 == 69 )
      {
        ++a1;
        v22 = 0;
      }
      else
      {
        v22 = 0;
        if ( !*(_DWORD *)(a4 + 24) )
        {
          ++*(_QWORD *)(a4 + 48);
          *(_DWORD *)(a4 + 24) = 1;
          *(_QWORD *)(a4 + 32) = v46;
        }
      }
    }
    else if ( (_BYTE)result == 111 )
    {
      a1 += 2;
      v21 = 0;
      v22 = " noexcept";
    }
    else if ( *(_DWORD *)(a4 + 24) )
    {
LABEL_39:
      v21 = 0;
      v22 = 0;
    }
    else
    {
      ++*(_QWORD *)(a4 + 32);
      v21 = 0;
      v22 = 0;
      ++*(_QWORD *)(a4 + 48);
      *(_DWORD *)(a4 + 24) = 1;
    }
    if ( a3 && !*(_QWORD *)(a4 + 32) )
    {
      v23 = *(_QWORD *)(a4 + 8);
      v24 = v23 + 1;
      if ( !*(_DWORD *)(a4 + 28) )
      {
        v44 = *(_QWORD *)(a4 + 16);
        if ( v44 > v24 )
        {
          *(_BYTE *)(*(_QWORD *)a4 + v23) = 41;
          v24 = *(_QWORD *)(a4 + 8) + 1LL;
        }
        else
        {
          *(_DWORD *)(a4 + 28) = 1;
          if ( v44 )
          {
            *(_BYTE *)(*(_QWORD *)a4 + v44 - 1) = 0;
            v24 = *(_QWORD *)(a4 + 8) + 1LL;
          }
        }
      }
      *(_QWORD *)(a4 + 8) = v24;
    }
    v25 = a1 + 2;
    if ( a1[1] != 89 )
      v25 = a1 + 1;
    ++*(_QWORD *)(a4 + 48);
    v53 = v10;
    v26 = (char *)sub_8EBA20(v25, 0, 2, a4);
    v27 = *(_QWORD *)(a4 + 48);
    *(_QWORD *)(a4 + 48) = v27 - 1;
    v28 = *v26;
    if ( *v26 == 82 )
    {
      v28 = v26[1];
      v29 = 1;
    }
    else
    {
      v29 = 0;
      if ( v28 == 79 )
      {
        v28 = v26[1];
        v29 = 2;
      }
    }
    if ( v28 != 69 && !*(_DWORD *)(a4 + 24) )
    {
      ++*(_QWORD *)(a4 + 32);
      *(_DWORD *)(a4 + 24) = 1;
      *(_QWORD *)(a4 + 48) = v27;
    }
    if ( v53 )
    {
      if ( !*(_QWORD *)(a4 + 32) )
      {
        v30 = *(_QWORD *)(a4 + 8);
        v31 = v30 + 1;
        if ( !*(_DWORD *)(a4 + 28) )
        {
          v47 = *(_QWORD *)(a4 + 16);
          if ( v47 > v31 )
          {
            *(_BYTE *)(*(_QWORD *)a4 + v30) = 32;
            v31 = *(_QWORD *)(a4 + 8) + 1LL;
          }
          else
          {
            *(_DWORD *)(a4 + 28) = 1;
            if ( v47 )
            {
              *(_BYTE *)(*(_QWORD *)a4 + v47 - 1) = 0;
              v31 = *(_QWORD *)(a4 + 8) + 1LL;
            }
          }
        }
        *(_QWORD *)(a4 + 8) = v31;
      }
      sub_8E6E80(v53, 0, a4);
    }
    if ( v29 && !*(_QWORD *)(a4 + 32) )
    {
      v32 = *(_QWORD *)(a4 + 8);
      v33 = v32 + 1;
      if ( *(_DWORD *)(a4 + 28) )
        goto LABEL_93;
      v34 = *(_QWORD *)(a4 + 16);
      if ( v34 <= v33 )
      {
        *(_DWORD *)(a4 + 28) = 1;
        if ( v34 )
        {
          *(_BYTE *)(*(_QWORD *)a4 + v34 - 1) = 0;
          v35 = *(_QWORD *)(a4 + 32);
          v36 = *(_QWORD *)(a4 + 8) + 1LL;
          goto LABEL_63;
        }
LABEL_93:
        *(_QWORD *)(a4 + 8) = v33;
        if ( v29 == 1 )
        {
LABEL_94:
          sub_8E5790((unsigned __int8 *)"&", a4);
          goto LABEL_65;
        }
LABEL_119:
        sub_8E5790((unsigned __int8 *)"&&", a4);
        goto LABEL_65;
      }
      *(_BYTE *)(*(_QWORD *)a4 + v32) = 32;
      v35 = *(_QWORD *)(a4 + 32);
      v36 = *(_QWORD *)(a4 + 8) + 1LL;
LABEL_63:
      *(_QWORD *)(a4 + 8) = v36;
      if ( v29 != 1 )
      {
        if ( v35 )
          goto LABEL_65;
        goto LABEL_119;
      }
      if ( !v35 )
        goto LABEL_94;
    }
LABEL_65:
    result = sub_8EB260(v25, 0, 0, a4);
    if ( v22 )
    {
      if ( !*(_QWORD *)(a4 + 32) )
        return sub_8E5790(v22, a4);
    }
    else if ( v21 )
    {
      if ( !*(_QWORD *)(a4 + 32) )
        sub_8E5790(" noexcept(", a4);
      result = (unsigned __int64)sub_8E74B0(v21, a4);
      if ( !*(_QWORD *)(a4 + 32) )
      {
        v38 = *(_QWORD *)(a4 + 8);
        result = v38 + 1;
        if ( !*(_DWORD *)(a4 + 28) )
        {
          v39 = *(_QWORD *)(a4 + 16);
          if ( v39 > result )
          {
            *(_BYTE *)(*(_QWORD *)a4 + v38) = 41;
            result = *(_QWORD *)(a4 + 8) + 1LL;
          }
          else
          {
            *(_DWORD *)(a4 + 28) = 1;
            if ( v39 )
            {
              *(_BYTE *)(*(_QWORD *)a4 + v39 - 1) = 0;
              result = *(_QWORD *)(a4 + 8) + 1LL;
            }
          }
        }
        *(_QWORD *)(a4 + 8) = result;
      }
    }
  }
  return result;
}
