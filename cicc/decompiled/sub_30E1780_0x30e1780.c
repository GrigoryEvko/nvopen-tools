// Function: sub_30E1780
// Address: 0x30e1780
//
char __fastcall sub_30E1780(__int64 a1, __int64 *a2, __int64 *a3)
{
  __int64 v3; // r9
  __int64 v4; // rcx
  __int64 v5; // rsi
  __int64 *v6; // rdi
  int v7; // r8d
  unsigned int v8; // eax
  __int64 *v9; // rbx
  __int64 v10; // rdx
  unsigned int v11; // eax
  __int64 *v12; // rdx
  __int64 v13; // r9
  int v14; // edi
  int v15; // esi
  char result; // al
  int v17; // r8d
  char v18; // cl
  __int64 *v19; // r14
  int v20; // r10d
  int v21; // edx
  int v22; // r10d
  char v23; // [rsp+Fh] [rbp-41h]
  char v24; // [rsp+Fh] [rbp-41h]
  unsigned __int64 v25; // [rsp+10h] [rbp-40h] BYREF
  unsigned int v26; // [rsp+18h] [rbp-38h]
  unsigned __int64 v27; // [rsp+20h] [rbp-30h] BYREF
  unsigned int v28; // [rsp+28h] [rbp-28h]

  v3 = *a2;
  v4 = *a3;
  v5 = *(unsigned int *)(*(_QWORD *)a1 + 240LL);
  v6 = *(__int64 **)(*(_QWORD *)a1 + 224LL);
  if ( (_DWORD)v5 )
  {
    v7 = v5 - 1;
    v8 = (v5 - 1) & (((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4));
    v9 = &v6[7 * v8];
    v10 = *v9;
    if ( v3 != *v9 )
    {
      v20 = 1;
      while ( v10 != -4096 )
      {
        v8 = v7 & (v20 + v8);
        v9 = &v6[7 * v8];
        v10 = *v9;
        if ( v3 == *v9 )
          goto LABEL_3;
        ++v20;
      }
      v9 = &v6[7 * (unsigned int)v5];
    }
LABEL_3:
    v11 = v7 & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
    v12 = &v6[7 * v11];
    v13 = *v12;
    if ( v4 != *v12 )
    {
      v21 = 1;
      while ( v13 != -4096 )
      {
        v22 = v21 + 1;
        v11 = v7 & (v21 + v11);
        v12 = &v6[7 * v11];
        v13 = *v12;
        if ( v4 == *v12 )
          goto LABEL_4;
        v21 = v22;
      }
      v12 = &v6[7 * v5];
    }
  }
  else
  {
    v12 = v6;
    v9 = v6;
  }
LABEL_4:
  v14 = *((_DWORD *)v12 + 2);
  v15 = *((_DWORD *)v9 + 2);
  result = v14 + *((_DWORD *)v12 + 3) < (int)qword_5030F68;
  v17 = v15 + *((_DWORD *)v9 + 3);
  if ( v14 + *((_DWORD *)v12 + 3) < (int)qword_5030F68 || (int)qword_5030F68 > v17 )
  {
    if ( result == (int)qword_5030F68 > v17 )
      return v14 < v15;
  }
  else
  {
    result = *((_BYTE *)v12 + 48);
    v18 = *((_BYTE *)v9 + 48);
    if ( !result && !v18 )
      return v14 < v15;
    if ( result == v18 )
    {
      v19 = v12 + 2;
      sub_C472A0((__int64)&v25, (__int64)(v12 + 4), v9 + 2);
      sub_C472A0((__int64)&v27, (__int64)(v9 + 4), v19);
      result = (int)sub_C49970((__int64)&v25, &v27) > 0;
      if ( v28 > 0x40 && v27 )
      {
        v23 = result;
        j_j___libc_free_0_0(v27);
        result = v23;
      }
      if ( v26 > 0x40 )
      {
        if ( v25 )
        {
          v24 = result;
          j_j___libc_free_0_0(v25);
          return v24;
        }
      }
    }
  }
  return result;
}
