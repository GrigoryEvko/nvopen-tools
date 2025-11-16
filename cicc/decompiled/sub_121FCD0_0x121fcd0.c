// Function: sub_121FCD0
// Address: 0x121fcd0
//
__int64 __fastcall sub_121FCD0(_QWORD *a1, __int64 a2, __int64 a3)
{
  signed int v4; // r15d
  signed int v5; // ecx
  bool v6; // al
  unsigned int v7; // eax
  __int64 v8; // rax
  __int64 v9; // r14
  __int64 result; // rax
  size_t v11; // r14
  size_t v12; // r8
  size_t v13; // rdx
  unsigned int v14; // eax
  unsigned int v15; // eax
  size_t v16; // r14
  size_t v17; // rcx
  size_t v18; // rdx
  unsigned int v19; // eax
  __int64 v20; // r14
  __int64 v21; // rcx
  signed int v22; // eax
  __int64 v23; // rbx
  unsigned int v24; // eax
  size_t v25; // r15
  size_t v26; // rcx
  size_t v27; // rdx
  int v28; // eax
  __int64 v29; // r15
  size_t v30; // r14
  size_t v31; // r15
  size_t v32; // rdx
  unsigned int v33; // eax
  __int64 v34; // r14
  __int64 v35; // r14
  size_t v36; // r14
  size_t v37; // r15
  size_t v38; // rdx
  unsigned int v39; // eax
  __int64 v40; // r14
  size_t v41; // [rsp+0h] [rbp-40h]
  signed int v42; // [rsp+8h] [rbp-38h]
  size_t v43; // [rsp+8h] [rbp-38h]
  size_t v44; // [rsp+8h] [rbp-38h]
  __int64 v45; // [rsp+8h] [rbp-38h]

  if ( (_QWORD *)a2 == a1 + 1 )
  {
    if ( !a1[5] )
      return sub_121FAD0((__int64)a1, a3);
    v23 = a1[4];
    v24 = *(_DWORD *)(v23 + 32);
    if ( v24 == *(_DWORD *)a3 )
    {
      if ( v24 <= 1 )
      {
        LOBYTE(v24) = *(_DWORD *)(v23 + 48) < *(_DWORD *)(a3 + 16);
      }
      else
      {
        v36 = *(_QWORD *)(v23 + 72);
        v38 = *(_QWORD *)(a3 + 40);
        v37 = v38;
        if ( v36 <= v38 )
          v38 = *(_QWORD *)(v23 + 72);
        if ( v38 && (v39 = memcmp(*(const void **)(v23 + 64), *(const void **)(a3 + 32), v38)) != 0 )
        {
          v24 = v39 >> 31;
        }
        else
        {
          v40 = v36 - v37;
          if ( v40 > 0x7FFFFFFF )
            return sub_121FAD0((__int64)a1, a3);
          if ( v40 < (__int64)0xFFFFFFFF80000000LL )
            return 0;
          LOBYTE(v24) = (int)v40 < 0;
        }
      }
    }
    else
    {
      LOBYTE(v24) = (int)v24 < *(_DWORD *)a3;
    }
    if ( !(_BYTE)v24 )
      return sub_121FAD0((__int64)a1, a3);
    return 0;
  }
  v4 = *(_DWORD *)a3;
  v5 = *(_DWORD *)(a2 + 32);
  if ( *(_DWORD *)a3 != v5 )
  {
    v6 = v4 < v5;
    goto LABEL_4;
  }
  if ( (unsigned int)v4 <= 1 )
  {
    v6 = *(_DWORD *)(a3 + 16) < *(_DWORD *)(a2 + 48);
LABEL_4:
    if ( !v6 )
    {
      if ( v4 != v5 )
      {
        LOBYTE(v7) = v5 < v4;
        goto LABEL_7;
      }
LABEL_21:
      if ( (unsigned int)v5 <= 1 )
      {
        LOBYTE(v7) = *(_DWORD *)(a2 + 48) < *(_DWORD *)(a3 + 16);
      }
      else
      {
        v16 = *(_QWORD *)(a2 + 72);
        v17 = *(_QWORD *)(a3 + 40);
        v18 = v17;
        if ( v16 <= v17 )
          v18 = *(_QWORD *)(a2 + 72);
        if ( v18
          && (v43 = *(_QWORD *)(a3 + 40),
              v19 = memcmp(*(const void **)(a2 + 64), *(const void **)(a3 + 32), v18),
              v17 = v43,
              v19) )
        {
          v7 = v19 >> 31;
        }
        else
        {
          v20 = v16 - v17;
          if ( v20 > 0x7FFFFFFF )
            return a2;
          if ( v20 < (__int64)0xFFFFFFFF80000000LL )
          {
LABEL_8:
            if ( a1[4] != a2 )
            {
              v8 = sub_220EEE0(a2);
              v9 = v8;
              if ( v4 != *(_DWORD *)(v8 + 32) )
              {
                if ( v4 < *(_DWORD *)(v8 + 32) )
                  goto LABEL_11;
                return sub_121FAD0((__int64)a1, a3);
              }
              if ( (unsigned int)v4 <= 1 )
              {
                if ( *(_DWORD *)(a3 + 16) >= *(_DWORD *)(v8 + 48) )
                  return sub_121FAD0((__int64)a1, a3);
                goto LABEL_11;
              }
              v25 = *(_QWORD *)(a3 + 40);
              v26 = *(_QWORD *)(v8 + 72);
              v27 = v26;
              if ( v25 <= v26 )
                v27 = *(_QWORD *)(a3 + 40);
              if ( !v27
                || (v44 = *(_QWORD *)(v8 + 72),
                    v28 = memcmp(*(const void **)(a3 + 32), *(const void **)(v8 + 64), v27),
                    v26 = v44,
                    !v28) )
              {
                v29 = v25 - v26;
                if ( v29 > 0x7FFFFFFF )
                  return sub_121FAD0((__int64)a1, a3);
                if ( v29 < (__int64)0xFFFFFFFF80000000LL )
                {
LABEL_11:
                  result = 0;
                  if ( *(_QWORD *)(a2 + 24) )
                    return v9;
                  return result;
                }
                v28 = v29;
              }
              if ( v28 >= 0 )
                return sub_121FAD0((__int64)a1, a3);
              goto LABEL_11;
            }
            return 0;
          }
          LOBYTE(v7) = (int)v20 < 0;
        }
      }
LABEL_7:
      if ( (_BYTE)v7 )
        goto LABEL_8;
      return a2;
    }
    goto LABEL_29;
  }
  v11 = *(_QWORD *)(a3 + 40);
  v12 = *(_QWORD *)(a2 + 72);
  v13 = v12;
  if ( v11 <= v12 )
    v13 = v11;
  if ( v13 )
  {
    v41 = *(_QWORD *)(a2 + 72);
    v42 = *(_DWORD *)(a2 + 32);
    v14 = memcmp(*(const void **)(a3 + 32), *(const void **)(a2 + 64), v13);
    v5 = v42;
    v12 = v41;
    if ( v14 )
    {
      v15 = v14 >> 31;
      goto LABEL_20;
    }
  }
  v35 = v11 - v12;
  if ( v35 > 0x7FFFFFFF )
    goto LABEL_21;
  if ( v35 >= (__int64)0xFFFFFFFF80000000LL )
  {
    LOBYTE(v15) = (int)v35 < 0;
LABEL_20:
    if ( !(_BYTE)v15 )
      goto LABEL_21;
  }
LABEL_29:
  result = a2;
  if ( a1[3] == a2 )
    return result;
  v21 = sub_220EF80(a2);
  v22 = *(_DWORD *)(v21 + 32);
  if ( v4 == v22 )
  {
    if ( (unsigned int)v4 <= 1 )
    {
      LOBYTE(v22) = *(_DWORD *)(v21 + 48) < *(_DWORD *)(a3 + 16);
    }
    else
    {
      v30 = *(_QWORD *)(v21 + 72);
      v31 = *(_QWORD *)(a3 + 40);
      v32 = v31;
      if ( v30 <= v31 )
        v32 = *(_QWORD *)(v21 + 72);
      if ( v32 && (v45 = v21, v33 = memcmp(*(const void **)(v21 + 64), *(const void **)(a3 + 32), v32), v21 = v45, v33) )
      {
        v22 = v33 >> 31;
      }
      else
      {
        v34 = v30 - v31;
        if ( v34 > 0x7FFFFFFF )
          return sub_121FAD0((__int64)a1, a3);
        if ( v34 < (__int64)0xFFFFFFFF80000000LL )
          goto LABEL_33;
        LOBYTE(v22) = (int)v34 < 0;
      }
    }
  }
  else
  {
    LOBYTE(v22) = v22 < v4;
  }
  if ( !(_BYTE)v22 )
    return sub_121FAD0((__int64)a1, a3);
LABEL_33:
  result = 0;
  if ( *(_QWORD *)(v21 + 24) )
    return a2;
  return result;
}
