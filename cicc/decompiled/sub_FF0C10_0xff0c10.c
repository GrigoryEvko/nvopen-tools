// Function: sub_FF0C10
// Address: 0xff0c10
//
__int64 __fastcall sub_FF0C10(__int64 a1, __int64 a2)
{
  __int64 v4; // rax
  int v5; // edx
  int v6; // ecx
  __int64 v7; // r8
  unsigned int v8; // edx
  _QWORD *v9; // r13
  __int64 v10; // rdi
  __int64 v11; // rax
  __int64 v12; // rdx
  bool v13; // zf
  bool v14; // si
  bool v15; // dl
  __int64 v16; // rdx
  __int64 v17; // rcx
  bool v18; // al
  bool v19; // zf
  __int64 v20; // rdx
  __int64 v21; // rcx
  __int64 result; // rax
  unsigned int v23; // esi
  int v24; // edi
  unsigned __int64 i; // r8
  int v26; // r14d
  __int64 v27; // r10
  unsigned int v28; // eax
  int v29; // r9d
  bool v30; // al
  _QWORD v31[2]; // [rsp+8h] [rbp-78h] BYREF
  __int64 v32; // [rsp+18h] [rbp-68h]
  __int64 v33; // [rsp+20h] [rbp-60h]
  void *v34; // [rsp+30h] [rbp-50h]
  _QWORD v35[2]; // [rsp+38h] [rbp-48h] BYREF
  __int64 v36; // [rsp+48h] [rbp-38h]
  __int64 v37; // [rsp+50h] [rbp-30h]

  v31[0] = 2;
  v31[1] = 0;
  v32 = a2;
  if ( a2 == -4096 || a2 == 0 || a2 == -8192 )
  {
    v4 = a2;
  }
  else
  {
    sub_BD73F0((__int64)v31);
    v4 = v32;
  }
  v33 = a1;
  v5 = *(_DWORD *)(a1 + 24);
  if ( v5 )
  {
    v6 = v5 - 1;
    v7 = *(_QWORD *)(a1 + 8);
    v8 = (v5 - 1) & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
    v9 = (_QWORD *)(v7 + 40LL * v8);
    v10 = v9[3];
    if ( v4 == v10 )
    {
LABEL_7:
      v36 = -8192;
      v34 = &unk_49DE380;
      v37 = 0;
      v11 = v9[3];
      v35[0] = 2;
      v35[1] = 0;
      if ( v11 == -8192 )
      {
        v9[4] = 0;
LABEL_14:
        --*(_DWORD *)(a1 + 16);
        v4 = v32;
        ++*(_DWORD *)(a1 + 20);
        goto LABEL_15;
      }
      if ( v11 == -4096 || !v11 )
      {
        v9[3] = -8192;
        v16 = v36;
        v17 = v37;
        v18 = v36 != -8192;
        v19 = v36 == -4096;
      }
      else
      {
        sub_BD60C0(v9 + 1);
        v12 = v36;
        v13 = v36 == -4096;
        v9[3] = v36;
        v14 = v12 != 0;
        v15 = v12 != -8192;
        if ( !v14 || v13 || !v15 )
        {
          v17 = v37;
          v30 = v14 && v15 && !v13;
LABEL_32:
          v9[4] = v17;
          v34 = &unk_49DB368;
          if ( v30 )
            sub_BD60C0(v35);
          goto LABEL_14;
        }
        sub_BD6050(v9 + 1, v35[0] & 0xFFFFFFFFFFFFFFF8LL);
        v16 = v36;
        v17 = v37;
        v18 = v36 != -4096;
        v19 = v36 == -8192;
      }
      v30 = v16 != 0 && !v19 && v18;
      goto LABEL_32;
    }
    v29 = 1;
    while ( v10 != -4096 )
    {
      v8 = v6 & (v29 + v8);
      v9 = (_QWORD *)(v7 + 40LL * v8);
      v10 = v9[3];
      if ( v4 == v10 )
        goto LABEL_7;
      ++v29;
    }
  }
LABEL_15:
  if ( v4 != 0 && v4 != -4096 && v4 != -8192 )
    sub_BD60C0(v31);
  v20 = *(unsigned int *)(a1 + 56);
  v21 = *(_QWORD *)(a1 + 40);
  result = (unsigned int)a2 >> 9;
  v23 = 0;
  v24 = 0;
  for ( i = (unsigned __int64)((unsigned int)result ^ ((unsigned int)a2 >> 4)) << 32; (_DWORD)v20; ++*(_DWORD *)(a1 + 52) )
  {
    v26 = 1;
    for ( result = ((_DWORD)v20 - 1)
                 & ((unsigned int)((0xBF58476D1CE4E5B9LL * (i | v23)) >> 31)
                  ^ (484763065 * ((unsigned int)i | v23))); ; result = ((_DWORD)v20 - 1) & v28 )
    {
      v27 = v21 + 24LL * (unsigned int)result;
      if ( a2 == *(_QWORD *)v27 && v24 == *(_DWORD *)(v27 + 8) )
        break;
      if ( *(_QWORD *)v27 == -4096 && *(_DWORD *)(v27 + 8) == -1 )
        return result;
      v28 = v26 + result;
      ++v26;
    }
    v23 += 37;
    result = v21 + 24 * v20;
    if ( v27 == result )
      break;
    *(_QWORD *)v27 = -8192;
    ++v24;
    *(_DWORD *)(v27 + 8) = -2;
    v20 = *(unsigned int *)(a1 + 56);
    --*(_DWORD *)(a1 + 48);
    v21 = *(_QWORD *)(a1 + 40);
  }
  return result;
}
