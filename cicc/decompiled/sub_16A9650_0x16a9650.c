// Function: sub_16A9650
// Address: 0x16a9650
//
__int64 __fastcall sub_16A9650(__int64 a1, __int64 a2, char *a3, unsigned __int64 a4, unsigned __int8 a5)
{
  unsigned __int8 v7; // cl
  char *v8; // r15
  __int64 result; // rax
  unsigned int v10; // r13d
  char *v11; // rbx
  bool v12; // r12
  __int64 v13; // r8
  unsigned int v14; // ecx
  unsigned __int64 v15; // rax
  int v16; // eax
  unsigned int v17; // edx
  size_t v18; // r13
  void *v19; // rax
  unsigned int v20; // ecx
  char v21; // [rsp+Fh] [rbp-51h]
  unsigned __int64 v22; // [rsp+10h] [rbp-50h]
  __int64 v23; // [rsp+18h] [rbp-48h]
  __int64 v24; // [rsp+18h] [rbp-48h]
  unsigned int v25; // [rsp+20h] [rbp-40h]
  unsigned __int8 v26; // [rsp+20h] [rbp-40h]
  unsigned int v27; // [rsp+24h] [rbp-3Ch]
  unsigned __int8 v28; // [rsp+24h] [rbp-3Ch]
  unsigned __int64 v29; // [rsp+28h] [rbp-38h]

  v7 = a5;
  v21 = *a3;
  if ( ((*a3 - 43) & 0xFD) != 0 )
  {
    result = *(unsigned int *)(a1 + 8);
    v29 = a4;
    v8 = a3;
    if ( (unsigned int)result <= 0x40 )
      goto LABEL_3;
LABEL_29:
    v26 = a5;
    v28 = a5;
    v18 = 8 * (((unsigned __int64)(unsigned int)result + 63) >> 6);
    v19 = (void *)sub_2207820(v18);
    result = (__int64)memset(v19, 0, v18);
    a5 = v26;
    v7 = v28;
    *(_QWORD *)a1 = result;
    goto LABEL_4;
  }
  v8 = a3 + 1;
  v29 = a4 - 1;
  result = *(unsigned int *)(a1 + 8);
  if ( (unsigned int)result > 0x40 )
    goto LABEL_29;
LABEL_3:
  *(_QWORD *)a1 = 0;
LABEL_4:
  v10 = 4;
  if ( a5 != 16 )
  {
    v10 = 3;
    if ( a5 != 8 )
      v10 = a5 == 2;
  }
  v11 = &a3[a4];
  if ( v8 != v11 )
  {
    v12 = a5 == 16 || a5 == 36;
    v25 = v7;
    v27 = v7 - 11;
    v22 = v7;
    do
    {
      v16 = *v8;
      v17 = v16 - 48;
      if ( v12 )
      {
        v13 = v17;
        if ( v17 > 9 )
        {
          v13 = (unsigned int)(v16 - 55);
          if ( v16 - 65 > v27 )
          {
            v13 = (unsigned int)(v16 - 87);
            if ( v27 < v16 - 97 )
              v13 = 0xFFFFFFFFLL;
          }
        }
      }
      else
      {
        v13 = v17;
        if ( v25 <= v17 )
          v13 = 0xFFFFFFFFLL;
      }
      if ( v29 > 1 )
      {
        if ( v10 )
        {
          v14 = *(_DWORD *)(a1 + 8);
          if ( v14 > 0x40 )
          {
            v24 = v13;
            sub_16A7DC0((__int64 *)a1, v10);
            v13 = v24;
          }
          else
          {
            v15 = 0;
            if ( v10 != v14 )
              v15 = (*(_QWORD *)a1 << v10) & (0xFFFFFFFFFFFFFFFFLL >> -(char)v14);
            *(_QWORD *)a1 = v15;
          }
        }
        else
        {
          v23 = v13;
          sub_16A7A10(a1, v22);
          v13 = v23;
        }
      }
      ++v8;
      result = sub_16A7490(a1, v13);
    }
    while ( v8 != v11 );
  }
  if ( v21 == 45 )
  {
    v20 = *(_DWORD *)(a1 + 8);
    if ( v20 <= 0x40 )
      *(_QWORD *)a1 = ~*(_QWORD *)a1 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v20);
    else
      sub_16A8F40((__int64 *)a1);
    return sub_16A7400(a1);
  }
  return result;
}
