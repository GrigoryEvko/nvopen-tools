// Function: sub_1E05960
// Address: 0x1e05960
//
__int64 __fastcall sub_1E05960(__int64 a1, __int64 a2)
{
  __int64 v5; // rax
  __int64 *v6; // rbx
  _QWORD *v7; // r11
  __int64 v8; // r8
  __int64 *v9; // rsi
  __int64 *v10; // r8
  __int64 *v11; // r9
  __int64 v12; // r10
  __int64 *v13; // r11
  __int64 v14; // rcx
  __int64 v15; // rdi
  __int64 *v16; // rax
  __int64 v17; // rdx
  __int64 v18; // r14
  __int64 *v19; // rax
  __int64 v20; // rdi
  _QWORD *v21; // rbx
  _QWORD *v22; // r14
  __int64 v23; // r13
  __int64 v24; // rax
  __int64 v25; // rsi
  unsigned int v26; // ecx
  __int64 *v27; // rdx
  __int64 v28; // r9
  int v29; // edx
  int v30; // r10d
  __int64 v31; // [rsp-48h] [rbp-48h] BYREF
  _QWORD *v32; // [rsp-38h] [rbp-38h]
  _QWORD *v33; // [rsp-30h] [rbp-30h]

  if ( *(_QWORD *)(a1 + 64) != *(_QWORD *)(a2 + 64) )
    return 1;
  v5 = *(unsigned int *)(a1 + 8);
  if ( v5 == *(_DWORD *)(a2 + 8) )
  {
    v6 = *(__int64 **)a1;
    v7 = *(_QWORD **)a2;
    v8 = *(_QWORD *)a1 + 8 * v5;
    if ( v8 == *(_QWORD *)a1 )
      goto LABEL_24;
    while ( *v6 == *v7 )
    {
      ++v6;
      ++v7;
      if ( (__int64 *)v8 == v6 )
        goto LABEL_24;
    }
    if ( (__int64 *)v8 == v6 )
    {
LABEL_24:
      if ( *(_DWORD *)(a2 + 40) == *(_DWORD *)(a1 + 40) )
      {
        sub_1E058E0(&v31, (__int64 *)(a1 + 24));
        v21 = v32;
        v22 = v33;
        v23 = *(_QWORD *)(a1 + 32) + 16LL * *(unsigned int *)(a1 + 48);
        if ( (_QWORD *)v23 == v32 )
          return 0;
        while ( 1 )
        {
          v24 = *(unsigned int *)(a2 + 48);
          if ( !(_DWORD)v24 )
            break;
          v25 = *(_QWORD *)(a2 + 32);
          v26 = (v24 - 1) & (((unsigned int)*v21 >> 9) ^ ((unsigned int)*v21 >> 4));
          v27 = (__int64 *)(v25 + 16LL * v26);
          v28 = *v27;
          if ( *v21 != *v27 )
          {
            v29 = 1;
            while ( v28 != -8 )
            {
              v30 = v29 + 1;
              v26 = (v24 - 1) & (v29 + v26);
              v27 = (__int64 *)(v25 + 16LL * v26);
              v28 = *v27;
              if ( *v21 == *v27 )
                goto LABEL_28;
              v29 = v30;
            }
            return 1;
          }
LABEL_28:
          if ( v27 == (__int64 *)(v25 + 16 * v24) || (unsigned __int8)sub_1E04E80(v21[1], v27[1]) )
            return 1;
          for ( v21 += 2; v22 != v21; v21 += 2 )
          {
            if ( *v21 != -16 && *v21 != -8 )
              break;
          }
          if ( (_QWORD *)v23 == v21 )
            return 0;
        }
      }
    }
    else
    {
      v9 = v6;
      while ( 1 )
      {
        if ( v9 == sub_1E04730(v6, (__int64)v9, v9) )
        {
          v14 = *v9;
          v15 = v12;
          v16 = v13;
          v17 = 0;
          while ( 1 )
          {
            ++v16;
            v17 += v14 == v15;
            if ( v16 == v11 )
              break;
            v15 = *v16;
          }
          if ( !v17 )
            break;
          if ( v10 == v9 )
            break;
          v18 = *v9;
          v19 = v9;
          v20 = 0;
          while ( 1 )
          {
            ++v19;
            v20 += v14 == v18;
            if ( v10 == v19 )
              break;
            v18 = *v19;
          }
          if ( v17 != v20 )
            break;
        }
        if ( v10 == ++v9 )
          goto LABEL_24;
      }
    }
  }
  return 1;
}
