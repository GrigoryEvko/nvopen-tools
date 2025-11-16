// Function: sub_2260000
// Address: 0x2260000
//
char __fastcall sub_2260000(int *a1, _BYTE *a2)
{
  __int64 v3; // r14
  char *v4; // rax
  __int64 v5; // rdx
  __int64 v6; // rcx
  __int64 v7; // rbx
  __int64 v8; // r12
  char *v9; // r15
  int v10; // eax
  __int64 v11; // r9
  int v12; // r10d
  size_t v13; // rdx
  int v14; // ecx
  unsigned int i; // r8d
  const void *v16; // rsi
  int v17; // eax
  char result; // al
  unsigned int v19; // r8d
  size_t v20; // [rsp+8h] [rbp-58h]
  int v21; // [rsp+10h] [rbp-50h]
  int v22; // [rsp+14h] [rbp-4Ch]
  size_t na; // [rsp+18h] [rbp-48h]
  unsigned int n; // [rsp+18h] [rbp-48h]
  __int64 v25; // [rsp+20h] [rbp-40h]
  int v26; // [rsp+20h] [rbp-40h]
  int v27; // [rsp+28h] [rbp-38h]
  __int64 v28; // [rsp+28h] [rbp-38h]

  v3 = *((_QWORD *)a1 + 1);
  v22 = *a1;
  v4 = (char *)sub_BD5D20((__int64)a2);
  v6 = *(unsigned int *)(v3 + 24);
  v25 = *(_QWORD *)(v3 + 8);
  v7 = v25 + 16 * v6;
  v27 = *(_DWORD *)(v3 + 24);
  v8 = v7;
  if ( (_DWORD)v6 )
  {
    na = v5;
    v9 = v4;
    v10 = sub_C94890(v4, v5);
    v11 = v25;
    v12 = 1;
    v13 = na;
    v14 = v27 - 1;
    for ( i = (v27 - 1) & v10; ; i = v14 & v19 )
    {
      v8 = v11 + 16LL * i;
      v16 = *(const void **)v8;
      if ( *(_QWORD *)v8 == -1 )
        break;
      if ( v16 == (const void *)-2LL )
      {
        if ( v9 == (char *)-2LL )
          goto LABEL_8;
      }
      else if ( *(_QWORD *)(v8 + 8) == v13 )
      {
        v21 = v12;
        n = i;
        v26 = v14;
        v28 = v11;
        if ( !v13 )
          goto LABEL_8;
        v20 = v13;
        v17 = memcmp(v9, v16, v13);
        v13 = v20;
        v11 = v28;
        v14 = v26;
        i = n;
        v12 = v21;
        if ( !v17 )
          goto LABEL_8;
      }
      v19 = v12 + i;
      ++v12;
    }
    if ( v9 != (char *)-1LL )
      v8 = *(_QWORD *)(v3 + 8) + 16LL * *(unsigned int *)(v3 + 24);
  }
LABEL_8:
  result = v7 != v8;
  if ( *a2 == 3 )
  {
    if ( (a2[32] & 0xF) != 0 )
    {
      if ( (a2[32] & 0xF) == 6 && v7 == v8 && v22 )
        return 0;
    }
    else if ( v7 == v8 )
    {
      return v22 == 0;
    }
    return 1;
  }
  return result;
}
