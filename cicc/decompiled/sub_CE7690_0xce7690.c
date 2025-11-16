// Function: sub_CE7690
// Address: 0xce7690
//
bool __fastcall sub_CE7690(__int64 a1, const void *a2, size_t a3, __int64 a4, char a5)
{
  size_t v6; // rdx
  __int64 v8; // r13
  char *v9; // r12
  __int64 v10; // rdi
  bool result; // al
  __int64 v12; // rbx
  unsigned __int8 v13; // al
  bool v14; // dl
  __int64 *v15; // rsi
  __int64 v16; // rcx
  _BYTE *v17; // rcx
  unsigned int v18; // ecx
  unsigned int v19; // r12d
  __int64 v20; // r13
  __int64 v21; // rdx
  _BYTE *v22; // rdi
  const void *v23; // rax
  __int64 v24; // rdx
  unsigned __int8 v25; // al
  __int64 v26; // rdx
  __int64 v27; // rax
  __int64 v28; // r8
  __int64 v29; // rdx
  int v30; // [rsp+Ch] [rbp-74h]
  __int64 v31; // [rsp+10h] [rbp-70h]
  int v34; // [rsp+24h] [rbp-5Ch]
  __int64 v35; // [rsp+28h] [rbp-58h]
  unsigned int v38; // [rsp+3Ch] [rbp-44h]
  __int64 v39; // [rsp+40h] [rbp-40h]
  __int64 v40; // [rsp+48h] [rbp-38h]

  v6 = 0;
  v8 = *(_QWORD *)(a1 + 40);
  v9 = off_4C5D0E8;
  if ( off_4C5D0E8 )
    v6 = strlen(off_4C5D0E8);
  v10 = sub_BA8DC0(v8, (__int64)v9, v6);
  v35 = v10;
  result = 0;
  if ( v10 )
  {
    v30 = *(_DWORD *)(a4 + 8);
    v38 = 0;
    v34 = sub_B91A00(v10);
    if ( v34 )
    {
      while ( 1 )
      {
        v12 = sub_B91A10(v35, v38);
        v39 = v12 - 16;
        v13 = *(_BYTE *)(v12 - 16);
        v14 = (v13 & 2) != 0;
        v15 = (v13 & 2) != 0 ? *(__int64 **)(v12 - 32) : (__int64 *)(v39 - 8LL * ((v13 >> 2) & 0xF));
        v16 = *v15;
        if ( *v15 )
        {
          if ( *(_BYTE *)v16 == 1 )
          {
            v17 = *(_BYTE **)(v16 + 136);
            if ( *v17 <= 3u && (_BYTE *)a1 == v17 )
            {
              v18 = (v13 & 2) != 0 ? *(_DWORD *)(v12 - 24) : (*(_WORD *)(v12 - 16) >> 6) & 0xF;
              if ( v18 > 1 )
                break;
            }
          }
        }
LABEL_33:
        if ( v34 == ++v38 )
          return *(_DWORD *)(a4 + 8) != v30;
      }
      v19 = 2;
      v20 = 8;
      v40 = 16LL * ((v18 - 2) >> 1) + 24;
      while ( 1 )
      {
        if ( v14 )
          v21 = *(_QWORD *)(v12 - 32);
        else
          v21 = v39 - 8LL * ((v13 >> 2) & 0xF);
        v22 = *(_BYTE **)(v21 + v20);
        if ( *v22 )
          v22 = 0;
        v23 = (const void *)sub_B91420((__int64)v22);
        if ( v24 == a3 && (!a3 || !memcmp(v23, a2, a3)) )
        {
          v25 = *(_BYTE *)(v12 - 16);
          v26 = (v25 & 2) != 0 ? *(_QWORD *)(v12 - 32) : v39 - 8LL * ((v25 >> 2) & 0xF);
          v27 = sub_CE7680(v26 + 8LL * v19);
          v29 = *(unsigned int *)(a4 + 8);
          if ( v29 + 1 > (unsigned __int64)*(unsigned int *)(a4 + 12) )
          {
            v31 = v27;
            sub_C8D5F0(a4, (const void *)(a4 + 16), v29 + 1, 8u, v28, v29 + 1);
            v29 = *(unsigned int *)(a4 + 8);
            v27 = v31;
          }
          *(_QWORD *)(*(_QWORD *)a4 + 8 * v29) = v27;
          ++*(_DWORD *)(a4 + 8);
          if ( a5 )
            return a5;
        }
        v19 += 2;
        v20 += 16;
        if ( v20 == v40 )
          goto LABEL_33;
        v13 = *(_BYTE *)(v12 - 16);
        v14 = (v13 & 2) != 0;
      }
    }
    else
    {
      return *(_DWORD *)(a4 + 8) != v30;
    }
  }
  return result;
}
