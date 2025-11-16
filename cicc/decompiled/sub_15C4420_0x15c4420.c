// Function: sub_15C4420
// Address: 0x15c4420
//
__int64 __fastcall sub_15C4420(__int64 *a1, void *a2, __int64 a3, unsigned int a4, char a5)
{
  __int64 result; // rax
  __int64 v10; // rax
  _QWORD *v11; // r15
  size_t v12; // r13
  char *v13; // rbx
  char *v14; // rax
  int v15; // eax
  __int64 v16; // r11
  int v17; // r8d
  unsigned int v18; // r10d
  __int64 *v19; // r9
  __int64 v20; // rax
  size_t v21; // rdx
  __int64 v22; // rcx
  const void *v23; // rsi
  int v24; // eax
  __int64 v25; // [rsp+8h] [rbp-68h]
  __int64 *v26; // [rsp+10h] [rbp-60h]
  size_t v27; // [rsp+18h] [rbp-58h]
  size_t n; // [rsp+20h] [rbp-50h]
  unsigned int na; // [rsp+20h] [rbp-50h]
  __int64 v30; // [rsp+28h] [rbp-48h]
  int v31; // [rsp+28h] [rbp-48h]
  __int64 v32; // [rsp+30h] [rbp-40h]
  __int64 v33; // [rsp+30h] [rbp-40h]
  int v34; // [rsp+38h] [rbp-38h]
  __int64 v35; // [rsp+38h] [rbp-38h]
  int i; // [rsp+38h] [rbp-38h]

  if ( a4 )
  {
LABEL_4:
    v35 = *a1 + 560;
    v10 = sub_161E980(48, 0);
    v11 = (_QWORD *)v10;
    if ( v10 )
    {
      v12 = 8 * a3;
      sub_1623D80(v10, (_DWORD)a1, 6, a4, 0, 0, 0, 0);
      v11[3] = 0;
      v11[4] = 0;
      v11[5] = 0;
      if ( (unsigned __int64)(8 * a3) > 0x7FFFFFFFFFFFFFF8LL )
        sub_4262D8((__int64)"cannot create std::vector larger than max_size()");
      v13 = 0;
      if ( v12 )
      {
        v14 = (char *)sub_22077B0(v12);
        v13 = &v14[v12];
        v11[3] = v14;
        v11[5] = &v14[v12];
        memcpy(v14, a2, v12);
      }
      v11[4] = v13;
    }
    return sub_15C42B0((__int64)v11, a4, v35);
  }
  v30 = *a1;
  v32 = *(_QWORD *)(*a1 + 568);
  v34 = *(_DWORD *)(*a1 + 584);
  if ( !v34 )
    goto LABEL_3;
  n = 8 * a3;
  v15 = sub_15B1DB0((__int64 *)a2, (__int64)a2 + 8 * a3);
  v16 = v32;
  v17 = v34 - 1;
  v18 = (v34 - 1) & v15;
  v19 = (__int64 *)(v32 + 8LL * v18);
  v20 = *v19;
  if ( *v19 == -8 )
    goto LABEL_3;
  v21 = n;
  v22 = v30;
  for ( i = 1; ; ++i )
  {
    if ( v20 != -16 )
    {
      v23 = *(const void **)(v20 + 24);
      if ( (__int64)(*(_QWORD *)(v20 + 32) - (_QWORD)v23) >> 3 == a3 )
      {
        na = v18;
        v31 = v17;
        v33 = v16;
        if ( !v21 )
          break;
        v25 = v22;
        v26 = v19;
        v27 = v21;
        v24 = memcmp(a2, v23, v21);
        v21 = v27;
        v19 = v26;
        v22 = v25;
        v16 = v33;
        v17 = v31;
        v18 = na;
        if ( !v24 )
          break;
      }
    }
    v18 = v17 & (i + v18);
    v19 = (__int64 *)(v16 + 8LL * v18);
    v20 = *v19;
    if ( *v19 == -8 )
      goto LABEL_3;
  }
  if ( v19 == (__int64 *)(*(_QWORD *)(v22 + 568) + 8LL * *(unsigned int *)(v22 + 584)) || (result = *v19) == 0 )
  {
LABEL_3:
    result = 0;
    if ( !a5 )
      return result;
    goto LABEL_4;
  }
  return result;
}
