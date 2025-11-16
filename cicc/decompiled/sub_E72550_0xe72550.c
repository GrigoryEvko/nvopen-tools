// Function: sub_E72550
// Address: 0xe72550
//
__int64 __fastcall sub_E72550(__int64 a1, __int64 a2)
{
  __int64 v2; // rdx
  unsigned int v3; // ebx
  __int64 v4; // rax
  size_t *v5; // r8
  size_t v6; // r15
  const void *v7; // r13
  size_t v8; // r14
  __int64 result; // rax
  size_t *v10; // r10
  size_t v11; // rdx
  const void *v12; // r12
  int v13; // eax
  bool v14; // sf
  int v15; // edx
  int v16; // edx
  unsigned int v17; // [rsp+Ch] [rbp-54h]
  unsigned __int8 v18; // [rsp+10h] [rbp-50h]
  unsigned __int8 v19; // [rsp+11h] [rbp-4Fh]
  unsigned __int8 v20; // [rsp+12h] [rbp-4Eh]
  unsigned __int8 v21; // [rsp+13h] [rbp-4Dh]
  unsigned int v22; // [rsp+14h] [rbp-4Ch]
  unsigned __int8 v23; // [rsp+18h] [rbp-48h]
  unsigned __int8 v24; // [rsp+19h] [rbp-47h]
  unsigned __int8 v25; // [rsp+1Ah] [rbp-46h]
  unsigned __int8 v26; // [rsp+1Bh] [rbp-45h]
  unsigned int v27; // [rsp+1Ch] [rbp-44h]
  size_t n; // [rsp+20h] [rbp-40h]
  unsigned int v29; // [rsp+28h] [rbp-38h]
  unsigned int v30; // [rsp+2Ch] [rbp-34h]

  v2 = *(_QWORD *)(a2 + 16);
  v3 = *(_DWORD *)(a2 + 60);
  v29 = *(_DWORD *)(a2 + 64);
  v30 = *(_DWORD *)(a1 + 60);
  v26 = *(_BYTE *)(a2 + 80);
  v27 = *(_DWORD *)(a1 + 64);
  v24 = *(_BYTE *)(a2 + 81);
  v25 = *(_BYTE *)(a1 + 80);
  v22 = *(_DWORD *)(a2 + 84);
  v23 = *(_BYTE *)(a1 + 81);
  v21 = *(_BYTE *)(a2 + 88);
  v17 = *(_DWORD *)(a1 + 84);
  v20 = *(_BYTE *)(a2 + 89);
  v4 = *(_QWORD *)(a1 + 16);
  v19 = *(_BYTE *)(a1 + 88);
  v18 = *(_BYTE *)(a1 + 89);
  if ( v2 && (*(_BYTE *)(v2 + 8) & 1) != 0 )
  {
    v5 = *(size_t **)(v2 - 8);
    v6 = *v5;
    v7 = v5 + 3;
    if ( !v4 )
    {
      if ( v6 )
        return 1;
      goto LABEL_21;
    }
    if ( (*(_BYTE *)(v4 + 8) & 1) == 0 )
    {
      v8 = 0;
      goto LABEL_6;
    }
  }
  else
  {
    if ( !v4 || (*(_BYTE *)(v4 + 8) & 1) == 0 )
      goto LABEL_21;
    v7 = 0;
    v6 = 0;
  }
  v10 = *(size_t **)(v4 - 8);
  v11 = v6;
  v8 = *v10;
  v12 = v10 + 3;
  if ( *v10 <= v6 )
    v11 = *v10;
  if ( v11 )
  {
    n = v11;
    v13 = memcmp(v10 + 3, v7, v11);
    v14 = v13 < 0;
    if ( v13 )
    {
      result = 1;
      if ( v14 )
        return result;
    }
    else
    {
      if ( v8 == v6 )
      {
        v15 = memcmp(v7, v12, n);
        if ( !v15 )
          goto LABEL_21;
        goto LABEL_36;
      }
      result = 1;
      if ( v8 < v6 )
        return result;
    }
    v15 = memcmp(v7, v12, n);
    if ( !v15 )
    {
      if ( v8 != v6 )
        goto LABEL_20;
      goto LABEL_21;
    }
LABEL_36:
    result = 0;
    if ( v15 < 0 )
      return result;
    goto LABEL_21;
  }
LABEL_6:
  if ( v6 != v8 )
  {
    result = 1;
    if ( v6 > v8 )
      return result;
LABEL_20:
    result = 0;
    if ( v6 < v8 )
      return result;
  }
LABEL_21:
  result = 1;
  if ( v3 <= v30 )
  {
    result = 0;
    if ( v3 == v30 )
    {
      result = 1;
      if ( v29 <= v27 )
      {
        result = 0;
        if ( v29 == v27 )
        {
          result = 1;
          if ( v26 <= v25 )
          {
            result = 0;
            if ( v26 == v25 )
            {
              result = 1;
              if ( v24 <= v23 )
              {
                result = 0;
                if ( v24 == v23 )
                {
                  result = 1;
                  if ( v22 <= v17 )
                  {
                    result = 0;
                    if ( v22 == v17 )
                    {
                      v16 = v19;
                      result = 1;
                      if ( v21 <= v19 )
                      {
                        LOBYTE(result) = v20 > v18;
                        LOBYTE(v16) = v21 == v19;
                        return v16 & (unsigned int)result;
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  return result;
}
