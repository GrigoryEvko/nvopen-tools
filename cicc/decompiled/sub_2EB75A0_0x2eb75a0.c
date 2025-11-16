// Function: sub_2EB75A0
// Address: 0x2eb75a0
//
__int64 *__fastcall sub_2EB75A0(__int64 *a1)
{
  __int64 v2; // rbx
  __int64 v3; // rdx
  __int64 v4; // rcx
  __int64 v5; // r8
  __int64 v6; // r9
  __int64 v7; // rax
  __int64 v8; // rbx
  __int64 v9; // r13
  __int64 *result; // rax
  __int64 *v11; // rdi
  __int64 *v12; // r15
  __int64 *v13; // rbx
  _QWORD *v14; // rax
  __int64 v15; // rdi
  __int64 v16; // r12
  unsigned int v17; // esi
  int v18; // edx
  __int64 *v19; // rcx
  __int64 *v20; // r9
  int v21; // r8d
  __int64 v22; // rdi
  unsigned int v23; // esi
  __int64 *v24; // rdx
  __int64 v25; // r11
  int v26; // edx
  int v27; // ebx
  int v28; // r11d
  _QWORD *v29; // r10
  __int64 *v30; // rax
  __int64 v31; // [rsp+28h] [rbp-98h]
  __int64 v32; // [rsp+30h] [rbp-90h] BYREF
  __int64 *v33; // [rsp+38h] [rbp-88h] BYREF
  __int64 *v34; // [rsp+40h] [rbp-80h] BYREF
  int v35; // [rsp+48h] [rbp-78h]
  char v36; // [rsp+50h] [rbp-70h] BYREF

  v2 = *a1;
  if ( *(_BYTE *)(*a1 + 32) )
  {
    sub_C7D6A0(*(_QWORD *)(v2 + 8), 16LL * *(unsigned int *)(v2 + 24), 8);
    ++*(_QWORD *)v2;
    *(_QWORD *)(v2 + 8) = 0;
    *(_QWORD *)(v2 + 16) = 0;
    *(_DWORD *)(v2 + 24) = 0;
  }
  else
  {
    *(_QWORD *)v2 = 1;
    *(_QWORD *)(v2 + 8) = 0;
    *(_QWORD *)(v2 + 16) = 0;
    *(_DWORD *)(v2 + 24) = 0;
    *(_BYTE *)(v2 + 32) = 1;
  }
  sub_C7D6A0(0, 0, 8);
  v7 = *(_QWORD *)(a1[1] + 128);
  v8 = v7 + 320;
  v9 = *(_QWORD *)(v7 + 328);
  result = (__int64 *)&v33;
  v31 = v8;
  if ( v9 == v8 )
    return result;
LABEL_6:
  while ( 2 )
  {
    if ( !*(_DWORD *)sub_2EB5B40(a1[2], v9, v3, v4, v5, v6) )
    {
      sub_2EB5530(&v34, v9, *(_QWORD *)(a1[2] + 4128), v4, v5);
      v11 = v34;
      v12 = &v34[v35];
      if ( v12 == v34 )
        goto LABEL_16;
      v13 = v34;
      while ( 1 )
      {
        v16 = *a1;
        v4 = *v13;
        v17 = *(_DWORD *)(*a1 + 24);
        v32 = *v13;
        if ( !v17 )
          break;
        v6 = v17 - 1;
        v5 = *(_QWORD *)(v16 + 8);
        v3 = (unsigned int)v6 & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
        v14 = (_QWORD *)(v5 + 16 * v3);
        v15 = *v14;
        if ( v4 != *v14 )
        {
          v28 = 1;
          v29 = 0;
          while ( v15 != -4096 )
          {
            if ( v29 || v15 != -8192 )
              v14 = v29;
            v3 = (unsigned int)v6 & (v28 + (_DWORD)v3);
            v15 = *(_QWORD *)(v5 + 16LL * (unsigned int)v3);
            if ( v4 == v15 )
              goto LABEL_10;
            ++v28;
            v29 = v14;
            v14 = (_QWORD *)(v5 + 16LL * (unsigned int)v3);
          }
          if ( !v29 )
            v29 = v14;
          v18 = *(_DWORD *)(v16 + 16) + 1;
          v33 = v29;
          ++*(_QWORD *)v16;
          if ( 4 * v18 < 3 * v17 )
          {
            v4 = v17 >> 3;
            if ( v17 - *(_DWORD *)(v16 + 20) - v18 > (unsigned int)v4 )
              goto LABEL_36;
            goto LABEL_14;
          }
LABEL_13:
          v17 *= 2;
LABEL_14:
          sub_2EB73C0(v16, v17);
          sub_2EB5230(v16, &v32, &v33);
          v18 = *(_DWORD *)(v16 + 16) + 1;
LABEL_36:
          v30 = v33;
          *(_DWORD *)(v16 + 16) = v18;
          if ( *v30 != -4096 )
            --*(_DWORD *)(v16 + 20);
          v3 = v32;
          *((_DWORD *)v30 + 2) = 0;
          *v30 = v3;
        }
LABEL_10:
        if ( v12 == ++v13 )
        {
          v11 = v34;
LABEL_16:
          if ( v11 == (__int64 *)&v36 )
            goto LABEL_5;
          _libc_free((unsigned __int64)v11);
          v9 = *(_QWORD *)(v9 + 8);
          if ( v31 == v9 )
            goto LABEL_18;
          goto LABEL_6;
        }
      }
      v33 = 0;
      ++*(_QWORD *)v16;
      goto LABEL_13;
    }
LABEL_5:
    v9 = *(_QWORD *)(v9 + 8);
    if ( v31 != v9 )
      continue;
    break;
  }
LABEL_18:
  result = *(__int64 **)(a1[1] + 128);
  v19 = (__int64 *)result[41];
  v20 = result + 40;
  if ( result + 40 != v19 )
  {
    v21 = 0;
    do
    {
      ++v21;
      v22 = *(_QWORD *)(*a1 + 8);
      result = (__int64 *)*(unsigned int *)(*a1 + 24);
      if ( (_DWORD)result )
      {
        v23 = ((_DWORD)result - 1) & (((unsigned int)v19 >> 9) ^ ((unsigned int)v19 >> 4));
        v24 = (__int64 *)(v22 + 16LL * v23);
        v25 = *v24;
        if ( v19 == (__int64 *)*v24 )
        {
LABEL_22:
          result = (__int64 *)(v22 + 16LL * (_QWORD)result);
          if ( v24 != result )
            *((_DWORD *)v24 + 2) = v21;
        }
        else
        {
          v26 = 1;
          while ( v25 != -4096 )
          {
            v27 = v26 + 1;
            v23 = ((_DWORD)result - 1) & (v26 + v23);
            v24 = (__int64 *)(v22 + 16LL * v23);
            v25 = *v24;
            if ( v19 == (__int64 *)*v24 )
              goto LABEL_22;
            v26 = v27;
          }
        }
      }
      v19 = (__int64 *)v19[1];
    }
    while ( v20 != v19 );
  }
  return result;
}
