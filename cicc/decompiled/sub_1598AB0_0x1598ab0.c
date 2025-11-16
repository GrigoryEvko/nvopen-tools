// Function: sub_1598AB0
// Address: 0x1598ab0
//
__int64 __fastcall sub_1598AB0(__int64 a1, __int64 a2, __int64 **a3)
{
  __int64 result; // rax
  int v5; // r13d
  __int64 v6; // r9
  int v8; // r11d
  __int64 *v9; // r10
  unsigned int i; // r8d
  __int64 *v11; // r15
  __int64 v12; // r14
  unsigned int v13; // r8d
  bool v14; // al
  bool v15; // dl
  int v16; // eax
  __int64 v17; // rsi
  _QWORD *v18; // rax
  __int64 v19; // rsi
  _QWORD *v20; // rdx
  bool v21; // al
  const void *v22; // rax
  __int64 v23; // rdx
  size_t v24; // rdx
  int v25; // eax
  __int64 v26; // [rsp+8h] [rbp-58h]
  int v27; // [rsp+10h] [rbp-50h]
  __int64 v28; // [rsp+10h] [rbp-50h]
  unsigned int v29; // [rsp+1Ch] [rbp-44h]
  int v30; // [rsp+1Ch] [rbp-44h]
  __int64 *v31; // [rsp+20h] [rbp-40h]
  unsigned int v32; // [rsp+20h] [rbp-40h]
  int v33; // [rsp+28h] [rbp-38h]
  __int64 *v34; // [rsp+28h] [rbp-38h]

  result = *(unsigned int *)(a1 + 24);
  if ( !(_DWORD)result )
  {
    *a3 = 0;
    return result;
  }
  v5 = result - 1;
  v6 = *(_QWORD *)(a1 + 8);
  v8 = 1;
  v9 = 0;
  for ( i = (result - 1) & *(_DWORD *)a2; ; i = v5 & v13 )
  {
    v11 = (__int64 *)(v6 + 8LL * i);
    v12 = *v11;
    if ( *v11 == -16 )
      break;
    if ( v12 == -8 )
      goto LABEL_8;
    if ( *(_QWORD *)(a2 + 8) == *(_QWORD *)v12
      && *(unsigned __int8 *)(a2 + 16) == *(unsigned __int16 *)(v12 + 18)
      && *(_BYTE *)(a2 + 17) == *(_BYTE *)(v12 + 17) >> 1
      && *(_QWORD *)(a2 + 32) == (*(_DWORD *)(v12 + 20) & 0xFFFFFFF) )
    {
      v26 = v6;
      v27 = v8;
      v29 = i;
      v31 = v9;
      v33 = *(unsigned __int16 *)(a2 + 18);
      v14 = sub_1594520(*v11);
      v9 = v31;
      i = v29;
      v15 = v14;
      v16 = 0;
      v8 = v27;
      v6 = v26;
      if ( v15 )
      {
        v16 = sub_1594720(v12);
        v6 = v26;
        v8 = v27;
        i = v29;
        v9 = v31;
      }
      if ( v33 == v16 )
      {
        v17 = *(_QWORD *)(a2 + 32);
        if ( (_DWORD)v17 )
        {
          v18 = *(_QWORD **)(a2 + 24);
          v19 = (__int64)&v18[(unsigned int)(v17 - 1) + 1];
          v20 = (_QWORD *)(v12 - 24LL * (*(_DWORD *)(v12 + 20) & 0xFFFFFFF));
          while ( *v18 == *v20 )
          {
            ++v18;
            v20 += 3;
            if ( (_QWORD *)v19 == v18 )
              goto LABEL_27;
          }
        }
        else
        {
LABEL_27:
          v28 = v6;
          v30 = v8;
          v32 = i;
          v34 = v9;
          v21 = sub_1594700(v12);
          v9 = v34;
          i = v32;
          v8 = v30;
          v6 = v28;
          if ( v21 )
          {
            v22 = (const void *)sub_1594710(v12);
            v9 = v34;
            i = v32;
            v8 = v30;
            v6 = v28;
            if ( *(_QWORD *)(a2 + 48) == v23 )
            {
              v24 = 4 * v23;
              if ( !v24
                || (v25 = memcmp(*(const void **)(a2 + 40), v22, v24), v9 = v34, i = v32, v8 = v30, v6 = v28, !v25) )
              {
LABEL_29:
                *a3 = v11;
                return 1;
              }
            }
          }
          else if ( !*(_QWORD *)(a2 + 48) )
          {
            goto LABEL_29;
          }
        }
      }
      v12 = *v11;
      break;
    }
LABEL_6:
    v13 = v8 + i;
    ++v8;
  }
  if ( v12 != -8 )
  {
    if ( v12 == -16 && !v9 )
      v9 = v11;
    goto LABEL_6;
  }
LABEL_8:
  if ( !v9 )
    v9 = v11;
  *a3 = v9;
  return 0;
}
