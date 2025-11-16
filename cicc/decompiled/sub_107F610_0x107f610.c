// Function: sub_107F610
// Address: 0x107f610
//
unsigned __int64 __fastcall sub_107F610(__int64 a1, __int64 a2)
{
  __int64 v3; // rdi
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // rcx
  __int64 v7; // r8
  __int64 v8; // r9
  unsigned __int64 **v9; // r12
  unsigned __int64 **v10; // r13
  unsigned __int64 result; // rax
  unsigned __int64 *v12; // rbx
  unsigned int v13; // esi
  __int64 v14; // r15
  __int64 v15; // r9
  unsigned int v16; // r8d
  _QWORD *v17; // rax
  __int64 v18; // rdi
  const char *v19; // rdx
  const char *v20; // rax
  int v21; // r11d
  _QWORD *v22; // rdx
  int v23; // eax
  int v24; // edi
  int v25; // r8d
  int v26; // r8d
  __int64 v27; // r9
  __int64 v28; // r11
  int v29; // esi
  _QWORD *v30; // rcx
  int v31; // r9d
  int v32; // r9d
  __int64 v33; // r10
  int v34; // esi
  unsigned int v35; // ecx
  __int64 v36; // r8
  unsigned int v37; // [rsp+8h] [rbp-78h]
  __int64 i; // [rsp+10h] [rbp-70h]
  const char *v39[4]; // [rsp+20h] [rbp-60h] BYREF
  __int16 v40; // [rsp+40h] [rbp-40h]

  v39[0] = "__indirect_function_table";
  v3 = *(_QWORD *)a2;
  v40 = 259;
  v4 = sub_E65280(v3, v39);
  if ( v4 && *(char *)(v4 + 12) < 0 )
    sub_E5CB20(a2, v4, v5, v6, v7, v8);
  v9 = *(unsigned __int64 ***)(a2 + 56);
  v10 = &v9[*(unsigned int *)(a2 + 64)];
  result = a1 + 400;
  for ( i = a1 + 400; v10 != v9; ++v9 )
  {
    v12 = *v9;
    if ( !**v9 )
    {
      result = *((_BYTE *)v12 + 9) & 0x70;
      if ( (_BYTE)result != 32 )
        continue;
      if ( *((char *)v12 + 8) < 0 )
        continue;
      *((_BYTE *)v12 + 8) |= 8u;
      result = (unsigned __int64)sub_E807D0(v12[3]);
      *v12 = result;
      if ( !result )
        continue;
    }
    if ( !*((_BYTE *)v12 + 36) )
      continue;
    if ( *((_DWORD *)v12 + 8) )
      continue;
    result = *((_BYTE *)v12 + 9) & 0x70;
    if ( (_BYTE)result == 32 )
      continue;
    v13 = *(_DWORD *)(a1 + 424);
    v14 = *(_QWORD *)(*v12 + 8);
    if ( v13 )
    {
      v15 = *(_QWORD *)(a1 + 408);
      v16 = (v13 - 1) & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
      v17 = (_QWORD *)(v15 + 16LL * v16);
      v18 = *v17;
      if ( v14 == *v17 )
      {
LABEL_17:
        v19 = *(const char **)(v14 + 128);
        v20 = *(const char **)(v14 + 136);
        v40 = 1283;
        v39[0] = "section already has a defining function: ";
        v39[2] = v19;
        v39[3] = v20;
        sub_C64D30((__int64)v39, 1u);
      }
      v21 = 1;
      v22 = 0;
      while ( v18 != -4096 )
      {
        if ( v22 || v18 != -8192 )
          v17 = v22;
        v16 = (v13 - 1) & (v21 + v16);
        v18 = *(_QWORD *)(v15 + 16LL * v16);
        if ( v14 == v18 )
          goto LABEL_17;
        ++v21;
        v22 = v17;
        v17 = (_QWORD *)(v15 + 16LL * v16);
      }
      if ( !v22 )
        v22 = v17;
      v23 = *(_DWORD *)(a1 + 416);
      ++*(_QWORD *)(a1 + 400);
      v24 = v23 + 1;
      if ( 4 * (v23 + 1) < 3 * v13 )
      {
        result = v13 - *(_DWORD *)(a1 + 420) - v24;
        if ( (unsigned int)result <= v13 >> 3 )
        {
          v37 = ((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4);
          sub_107F430(i, v13);
          v31 = *(_DWORD *)(a1 + 424);
          if ( !v31 )
          {
LABEL_57:
            ++*(_DWORD *)(a1 + 416);
            BUG();
          }
          v32 = v31 - 1;
          v33 = *(_QWORD *)(a1 + 408);
          v34 = 1;
          v35 = v32 & v37;
          v24 = *(_DWORD *)(a1 + 416) + 1;
          result = 0;
          v22 = (_QWORD *)(v33 + 16LL * (v32 & v37));
          v36 = *v22;
          if ( v14 != *v22 )
          {
            while ( v36 != -4096 )
            {
              if ( !result && v36 == -8192 )
                result = (unsigned __int64)v22;
              v35 = v32 & (v34 + v35);
              v22 = (_QWORD *)(v33 + 16LL * v35);
              v36 = *v22;
              if ( v14 == *v22 )
                goto LABEL_25;
              ++v34;
            }
            if ( result )
              v22 = (_QWORD *)result;
          }
        }
        goto LABEL_25;
      }
    }
    else
    {
      ++*(_QWORD *)(a1 + 400);
    }
    sub_107F430(i, 2 * v13);
    v25 = *(_DWORD *)(a1 + 424);
    if ( !v25 )
      goto LABEL_57;
    v26 = v25 - 1;
    v27 = *(_QWORD *)(a1 + 408);
    result = v26 & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
    v24 = *(_DWORD *)(a1 + 416) + 1;
    v22 = (_QWORD *)(v27 + 16 * result);
    v28 = *v22;
    if ( v14 != *v22 )
    {
      v29 = 1;
      v30 = 0;
      while ( v28 != -4096 )
      {
        if ( v28 == -8192 && !v30 )
          v30 = v22;
        result = v26 & (unsigned int)(v29 + result);
        v22 = (_QWORD *)(v27 + 16LL * (unsigned int)result);
        v28 = *v22;
        if ( v14 == *v22 )
          goto LABEL_25;
        ++v29;
      }
      if ( v30 )
        v22 = v30;
    }
LABEL_25:
    *(_DWORD *)(a1 + 416) = v24;
    if ( *v22 != -4096 )
      --*(_DWORD *)(a1 + 420);
    *v22 = v14;
    v22[1] = v12;
  }
  return result;
}
