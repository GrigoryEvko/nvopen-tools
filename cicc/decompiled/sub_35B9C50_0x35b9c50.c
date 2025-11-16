// Function: sub_35B9C50
// Address: 0x35b9c50
//
unsigned __int64 __fastcall sub_35B9C50(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  unsigned __int64 v4; // rax
  unsigned int v5; // r14d
  unsigned int v6; // ebx
  _DWORD *v7; // rsi
  __int64 v8; // rcx
  __int64 v9; // rdx
  size_t v10; // r15
  void *v11; // rax
  __int64 v12; // rcx
  void *v13; // rdi
  void *v14; // rax
  unsigned __int64 v15; // r8
  _DWORD *v16; // rsi
  __int64 v17; // rax
  unsigned __int64 result; // rax
  _DWORD *v19; // rsi
  unsigned int v20; // r8d
  int v21; // r9d
  __int64 v22; // r14
  __int64 v23; // rbx
  __int64 v24; // rdx
  __int64 v25; // r10
  __int64 v26; // rsi
  __int64 v27; // rdi
  int v28; // ecx
  __int64 v29; // r11
  unsigned int i; // edx
  __int64 v31; // rcx
  unsigned int *v32; // rbx
  __int64 v33; // rdi
  __int64 v34; // rdx
  __int64 v35; // r11
  unsigned int v36; // edx
  __int64 v37; // rcx
  _DWORD *v38; // rsi
  bool v39; // zf
  int v40; // [rsp+Ch] [rbp-54h]
  __int64 v41; // [rsp+10h] [rbp-50h]
  __int64 v42; // [rsp+18h] [rbp-48h]
  int v43[13]; // [rsp+2Ch] [rbp-34h] BYREF

  v3 = *(_QWORD *)(a1 + 168) - *(_QWORD *)(a1 + 160);
  *(_QWORD *)(a1 + 152) = a2;
  v43[0] = 0;
  v4 = 0xAAAAAAAAAAAAAAABLL * (v3 >> 5);
  if ( (_DWORD)v4 )
  {
    v5 = v4;
    do
    {
      v7 = *(_DWORD **)(a1 + 192);
      if ( v7 == sub_35B8320(*(_DWORD **)(a1 + 184), (__int64)v7, v43) )
      {
        v6 = v43[0];
        v40 = -1431655765 * ((__int64)(*(_QWORD *)(a1 + 168) - *(_QWORD *)(a1 + 160)) >> 5);
        goto LABEL_6;
      }
      v6 = v43[0] + 1;
      v43[0] = v6;
    }
    while ( v5 > v6 );
    v40 = -1431655765 * ((__int64)(*(_QWORD *)(a1 + 168) - *(_QWORD *)(a1 + 160)) >> 5);
LABEL_6:
    while ( v40 != v6 )
    {
      while ( 1 )
      {
        v8 = *(_QWORD *)(**(_QWORD **)(a1 + 152) + 160LL) + 96LL * v6;
        v41 = v8;
        v9 = (unsigned int)(**(_DWORD **)v8 - 1);
        *(_DWORD *)(v8 + 20) = v9;
        v10 = 4 * v9;
        v42 = v9;
        v11 = (void *)sub_2207820(4 * v9);
        v12 = v41;
        v13 = v11;
        if ( v11 && v42 )
        {
          v14 = memset(v11, 0, v10);
          v12 = v41;
          v13 = v14;
        }
        v15 = *(_QWORD *)(v12 + 32);
        *(_QWORD *)(v12 + 32) = v13;
        if ( v15 )
          j_j___libc_free_0_0(v15);
        v43[0] = ++v6;
        if ( v5 <= v6 )
          break;
        while ( 1 )
        {
          v16 = *(_DWORD **)(a1 + 192);
          if ( v16 == sub_35B8320(*(_DWORD **)(a1 + 184), (__int64)v16, v43) )
            break;
          v6 = v43[0] + 1;
          v43[0] = v6;
          if ( v5 <= v6 )
            goto LABEL_6;
        }
        v6 = v43[0];
        if ( v40 == v43[0] )
          goto LABEL_17;
      }
    }
  }
LABEL_17:
  v17 = *(_QWORD *)(a1 + 216) - *(_QWORD *)(a1 + 208);
  v43[0] = 0;
  result = 0xAAAAAAAAAAAAAAABLL * (v17 >> 4);
  if ( (_DWORD)result )
  {
    do
    {
      v19 = *(_DWORD **)(a1 + 240);
      if ( v19 == sub_35B8320(*(_DWORD **)(a1 + 232), (__int64)v19, v43) )
      {
        result = (unsigned int)v43[0];
        v21 = -1431655765 * ((__int64)(*(_QWORD *)(a1 + 216) - *(_QWORD *)(a1 + 208)) >> 4);
        goto LABEL_22;
      }
      result = (unsigned int)(v43[0] + 1);
      v43[0] = result;
    }
    while ( v20 > (unsigned int)result );
    v21 = -1431655765 * ((__int64)(*(_QWORD *)(a1 + 216) - *(_QWORD *)(a1 + 208)) >> 4);
LABEL_22:
    while ( 2 )
    {
      if ( (_DWORD)result == v21 )
        return result;
      while ( 1 )
      {
        v22 = *(_QWORD *)(a1 + 152);
        v23 = 48LL * (unsigned int)result;
        v24 = v23 + *(_QWORD *)(*(_QWORD *)v22 + 208LL);
        v25 = *(_QWORD *)v24;
        v26 = *(unsigned int *)(v24 + 20);
        v27 = *(_QWORD *)(*(_QWORD *)v22 + 160LL) + 96 * v26;
        v28 = *(_DWORD *)(v27 + 24);
        if ( (_DWORD)v26 == *(_DWORD *)(v24 + 24) )
        {
          *(_DWORD *)(v27 + 24) = *(_DWORD *)(v25 + 16) + v28;
          v29 = *(_QWORD *)(v25 + 32);
        }
        else
        {
          *(_DWORD *)(v27 + 24) = *(_DWORD *)(v25 + 20) + v28;
          v29 = *(_QWORD *)(v25 + 24);
        }
        for ( i = 0;
              *(_DWORD *)(v27 + 20) > i;
              *(_DWORD *)(*(_QWORD *)(v27 + 32) + 4 * v31) += *(unsigned __int8 *)(v29 + v31) )
        {
          v31 = i++;
        }
        v32 = (unsigned int *)(*(_QWORD *)(*(_QWORD *)v22 + 208LL) + v23);
        v33 = *(_QWORD *)(*(_QWORD *)v22 + 160LL) + 96LL * v32[6];
        v34 = *(_QWORD *)v32;
        *(_DWORD *)(v33 + 24) += *(_DWORD *)(*(_QWORD *)v32 + 16LL);
        v35 = *(_QWORD *)(v34 + 32);
        if ( *(_DWORD *)(v33 + 20) )
          break;
        while ( 1 )
        {
          result = (unsigned int)(result + 1);
          v43[0] = result;
          if ( v20 <= (unsigned int)result )
            break;
LABEL_31:
          v38 = *(_DWORD **)(a1 + 240);
          v39 = v38 == sub_35B8320(*(_DWORD **)(a1 + 232), (__int64)v38, v43);
          result = (unsigned int)v43[0];
          if ( v39 )
            goto LABEL_22;
        }
        if ( (_DWORD)result == v21 )
          return result;
      }
      v36 = 0;
      do
      {
        v37 = v36++;
        *(_DWORD *)(*(_QWORD *)(v33 + 32) + 4 * v37) += *(unsigned __int8 *)(v35 + v37);
      }
      while ( *(_DWORD *)(v33 + 20) > v36 );
      result = (unsigned int)(result + 1);
      v43[0] = result;
      if ( v20 <= (unsigned int)result )
        continue;
      goto LABEL_31;
    }
  }
  return result;
}
