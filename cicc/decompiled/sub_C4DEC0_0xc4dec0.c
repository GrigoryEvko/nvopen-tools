// Function: sub_C4DEC0
// Address: 0xc4dec0
//
__int64 __fastcall sub_C4DEC0(__int64 a1, __int64 a2, unsigned int a3, char a4)
{
  unsigned int v6; // r12d
  unsigned int v7; // r15d
  unsigned int v9; // r12d
  unsigned int v10; // r15d
  unsigned int v11; // r14d
  bool v12; // al
  __int64 v13; // rdi
  bool v14; // al
  __int64 v15; // rcx
  unsigned int v16; // esi
  unsigned int v17; // r14d
  int v18; // eax
  int v19; // ebx
  unsigned __int64 v20; // rax
  __int64 v21; // rcx
  unsigned __int64 v22; // rdx
  unsigned int v23; // r15d
  unsigned __int64 v24; // rdi
  unsigned int v25; // [rsp+Ch] [rbp-74h]
  __int64 v26; // [rsp+10h] [rbp-70h]
  bool v27; // [rsp+10h] [rbp-70h]
  unsigned int v28; // [rsp+10h] [rbp-70h]
  bool v29; // [rsp+10h] [rbp-70h]
  unsigned __int64 v31; // [rsp+20h] [rbp-60h]
  unsigned __int64 v33; // [rsp+30h] [rbp-50h] BYREF
  unsigned int v34; // [rsp+38h] [rbp-48h]
  __int64 v35; // [rsp+40h] [rbp-40h] BYREF
  unsigned int v36; // [rsp+48h] [rbp-38h]

  v6 = *(_DWORD *)(a2 + 8);
  if ( a3 == v6 )
  {
    *(_DWORD *)(a1 + 8) = a3;
    if ( a3 > 0x40 )
      sub_C43780(a1, (const void **)a2);
    else
      *(_QWORD *)a1 = *(_QWORD *)a2;
    return a1;
  }
  v34 = a3;
  if ( a3 > 0x40 )
  {
    sub_C43690((__int64)&v33, 0, 0);
    v7 = *(_DWORD *)(a2 + 8);
    if ( v7 > 0x40 )
    {
LABEL_4:
      if ( (unsigned int)sub_C444A0(a2) == v7 )
      {
LABEL_5:
        *(_DWORD *)(a1 + 8) = v34;
        *(_QWORD *)a1 = v33;
        return a1;
      }
      goto LABEL_9;
    }
  }
  else
  {
    v7 = v6;
    v33 = 0;
    if ( v6 > 0x40 )
      goto LABEL_4;
  }
  if ( !*(_QWORD *)a2 )
    goto LABEL_5;
LABEL_9:
  if ( a3 <= v6 )
  {
    v9 = v6 / a3;
    v10 = 0;
    v11 = 0;
    while ( 1 )
    {
      if ( a4 )
      {
        sub_C440A0((__int64)&v35, (__int64 *)a2, v9, v10);
        if ( v36 )
        {
          if ( v36 <= 0x40 )
          {
            v12 = v35 == 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v36);
          }
          else
          {
            v25 = v36;
            v26 = v35;
            v12 = v25 == (unsigned int)sub_C445E0((__int64)&v35);
            if ( v26 )
            {
              v13 = v26;
              v27 = v12;
              j_j___libc_free_0_0(v13);
              v12 = v27;
            }
          }
          if ( !v12 )
            goto LABEL_17;
        }
      }
      else
      {
        sub_C440A0((__int64)&v35, (__int64 *)a2, v9, v10);
        if ( v36 <= 0x40 )
        {
          v14 = v35 == 0;
        }
        else
        {
          v28 = v36;
          v14 = v28 == (unsigned int)sub_C444A0((__int64)&v35);
          if ( v35 )
          {
            v29 = v14;
            j_j___libc_free_0_0(v35);
            v14 = v29;
          }
        }
        if ( v14 )
          goto LABEL_17;
      }
      v15 = 1LL << v11;
      if ( v34 > 0x40 )
      {
        *(_QWORD *)(v33 + 8LL * (v11 >> 6)) |= v15;
LABEL_17:
        ++v11;
        v10 += v9;
        if ( a3 == v11 )
          goto LABEL_25;
      }
      else
      {
        ++v11;
        v10 += v9;
        v33 |= v15;
        if ( a3 == v11 )
          goto LABEL_25;
      }
    }
  }
  v16 = 0;
  v17 = 0;
  v18 = a3 / v6;
  v19 = a3 / v6;
  v20 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v18);
  while ( 1 )
  {
    v21 = 1LL << v17;
    v22 = *(_QWORD *)a2;
    if ( v7 > 0x40 )
      v22 = *(_QWORD *)(v22 + 8LL * (v17 >> 6));
    ++v17;
    v23 = v16 + v19;
    if ( (v22 & v21) != 0 && v23 != v16 )
    {
      if ( v16 > 0x3F || v23 > 0x40 )
      {
        v31 = v20;
        sub_C43C90(&v33, v16, v23);
        v20 = v31;
      }
      else
      {
        v24 = v20 << v16;
        if ( v34 > 0x40 )
          *(_QWORD *)v33 |= v24;
        else
          v33 |= v24;
      }
    }
    v16 += v19;
    if ( v6 == v17 )
      break;
    v7 = *(_DWORD *)(a2 + 8);
  }
LABEL_25:
  *(_DWORD *)(a1 + 8) = v34;
  *(_QWORD *)a1 = v33;
  return a1;
}
