// Function: sub_B0D000
// Address: 0xb0d000
//
__int64 __fastcall sub_B0D000(__int64 *a1, __int64 *a2, __int64 a3, unsigned int a4, char a5)
{
  int v9; // eax
  int v10; // r9d
  size_t v11; // rdx
  __int64 v12; // r8
  int v13; // r9d
  unsigned int i; // r11d
  __int64 *v15; // r10
  __int64 v16; // rcx
  const void *v17; // rsi
  int v18; // eax
  unsigned int v19; // r11d
  __int64 result; // rax
  __int64 v21; // rax
  _QWORD *v22; // r15
  size_t v23; // r13
  char *v24; // rbx
  char *v25; // rax
  __int64 v26; // [rsp+0h] [rbp-70h]
  __int64 *v27; // [rsp+8h] [rbp-68h]
  __int64 v28; // [rsp+10h] [rbp-60h]
  size_t v29; // [rsp+18h] [rbp-58h]
  size_t na; // [rsp+20h] [rbp-50h]
  unsigned int n; // [rsp+20h] [rbp-50h]
  __int64 v32; // [rsp+28h] [rbp-48h]
  int v33; // [rsp+28h] [rbp-48h]
  int v34; // [rsp+34h] [rbp-3Ch]
  int v35; // [rsp+34h] [rbp-3Ch]
  __int64 v36; // [rsp+38h] [rbp-38h]
  __int64 v37; // [rsp+38h] [rbp-38h]

  if ( a4 )
    goto LABEL_11;
  v32 = *a1;
  v36 = *(_QWORD *)(*a1 + 736);
  v34 = *(_DWORD *)(*a1 + 752);
  if ( v34 )
  {
    na = 8 * a3;
    v9 = sub_AF66D0(a2, (__int64)&a2[a3]);
    v10 = v34;
    v35 = 1;
    v11 = na;
    v12 = v32;
    v13 = v10 - 1;
    for ( i = v13 & v9; ; i = v13 & v19 )
    {
      v15 = (__int64 *)(v36 + 8LL * i);
      v16 = *v15;
      if ( *v15 == -4096 )
        break;
      if ( v16 != -8192 )
      {
        v17 = *(const void **)(v16 + 16);
        if ( (__int64)(*(_QWORD *)(v16 + 24) - (_QWORD)v17) >> 3 == a3 )
        {
          n = i;
          v33 = v13;
          if ( !v11 )
            goto LABEL_17;
          v26 = *v15;
          v27 = (__int64 *)(v36 + 8LL * i);
          v28 = v12;
          v29 = v11;
          v18 = memcmp(a2, v17, v11);
          v11 = v29;
          v12 = v28;
          v15 = v27;
          v16 = v26;
          v13 = v33;
          i = n;
          if ( !v18 )
          {
LABEL_17:
            if ( v15 == (__int64 *)(*(_QWORD *)(v12 + 736) + 8LL * *(unsigned int *)(v12 + 752)) )
              break;
            return v16;
          }
        }
      }
      v19 = v35 + i;
      ++v35;
    }
  }
  result = 0;
  if ( a5 )
  {
LABEL_11:
    v37 = *a1 + 728;
    v21 = sub_B97910(40, 0, a4);
    v22 = (_QWORD *)v21;
    if ( v21 )
    {
      v23 = 8 * a3;
      sub_B971C0(v21, (_DWORD)a1, 7, a4, 0, 0, 0, 0);
      v22[2] = 0;
      v22[3] = 0;
      v22[4] = 0;
      if ( (unsigned __int64)(8 * a3) > 0x7FFFFFFFFFFFFFF8LL )
        sub_4262D8((__int64)"cannot create std::vector larger than max_size()");
      v24 = 0;
      if ( v23 )
      {
        v25 = (char *)sub_22077B0(v23);
        v24 = &v25[v23];
        v22[2] = v25;
        v22[4] = &v25[v23];
        memcpy(v25, a2, v23);
      }
      v22[3] = v24;
    }
    return sub_B0CE80((__int64)v22, a4, v37);
  }
  return result;
}
