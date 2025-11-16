// Function: sub_1C2E100
// Address: 0x1c2e100
//
bool __fastcall sub_1C2E100(__int64 a1, const void *a2, size_t a3, __int64 a4, char a5)
{
  __int64 v7; // rdi
  __int64 v8; // rdi
  bool result; // al
  __int64 v10; // rbx
  __int64 v11; // r13
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // r15
  _BYTE *v16; // rdi
  const void *v17; // rax
  __int64 v18; // rdx
  int v19; // r8d
  __int64 v20; // r9
  __int64 v21; // rax
  int v22; // [rsp+Ch] [rbp-84h]
  __int64 v23; // [rsp+10h] [rbp-80h]
  int v26; // [rsp+24h] [rbp-6Ch]
  __int64 v27; // [rsp+28h] [rbp-68h]
  unsigned int v29; // [rsp+3Ch] [rbp-54h]
  char *v30; // [rsp+40h] [rbp-50h] BYREF
  __int16 v31; // [rsp+50h] [rbp-40h]

  v31 = 257;
  v7 = *(_QWORD *)(a1 + 40);
  if ( *off_4CD4988 )
  {
    v30 = off_4CD4988;
    LOBYTE(v31) = 3;
  }
  v8 = sub_1632310(v7, (__int64)&v30);
  v27 = v8;
  result = 0;
  if ( v8 )
  {
    v22 = *(_DWORD *)(a4 + 8);
    v29 = 0;
    v26 = sub_161F520(v8);
    if ( v26 )
    {
      while ( 1 )
      {
        v10 = sub_161F530(v27, v29);
        v11 = *(unsigned int *)(v10 + 8);
        v12 = *(_QWORD *)(v10 - 8 * v11);
        if ( v12 )
        {
          if ( *(_BYTE *)v12 == 1 )
          {
            v13 = *(_QWORD *)(v12 + 136);
            if ( *(_BYTE *)(v13 + 16) <= 3u && a1 == v13 && (unsigned int)v11 > 1 )
              break;
          }
        }
LABEL_23:
        if ( v26 == ++v29 )
          return *(_DWORD *)(a4 + 8) != v22;
      }
      v14 = (unsigned int)v11;
      v15 = 1;
      while ( 1 )
      {
        v16 = *(_BYTE **)(v10 + 8 * (v15 - v14));
        if ( *v16 )
          v16 = 0;
        v17 = (const void *)sub_161E970((__int64)v16);
        if ( a3 == v18 && (!a3 || !memcmp(v17, a2, a3)) )
        {
          v20 = sub_1C2E0F0(v10 + 8 * ((unsigned int)(v15 + 1) - (unsigned __int64)*(unsigned int *)(v10 + 8)));
          v21 = *(unsigned int *)(a4 + 8);
          if ( (unsigned int)v21 >= *(_DWORD *)(a4 + 12) )
          {
            v23 = v20;
            sub_16CD150(a4, (const void *)(a4 + 16), 0, 8, v19, v20);
            v21 = *(unsigned int *)(a4 + 8);
            v20 = v23;
          }
          *(_QWORD *)(*(_QWORD *)a4 + 8 * v21) = v20;
          ++*(_DWORD *)(a4 + 8);
          if ( a5 )
            return a5;
        }
        v15 += 2;
        if ( (unsigned int)v11 <= (unsigned int)v15 )
          goto LABEL_23;
        v14 = *(unsigned int *)(v10 + 8);
      }
    }
    else
    {
      return *(_DWORD *)(a4 + 8) != v22;
    }
  }
  return result;
}
