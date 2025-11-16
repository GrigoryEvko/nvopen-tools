// Function: sub_227BED0
// Address: 0x227bed0
//
__int64 __fastcall sub_227BED0(__int64 a1)
{
  int v2; // r15d
  __int64 result; // rax
  _QWORD **v4; // rbx
  __int64 v5; // rdx
  _QWORD **v6; // r14
  _QWORD **v7; // r12
  __int64 v8; // rax
  _QWORD *v9; // rbx
  unsigned __int64 v10; // r15
  __int64 v11; // rdi
  int v12; // edx
  _QWORD **i; // rbx
  __int64 v14; // rax
  _QWORD *v15; // r12
  unsigned __int64 v16; // r8
  __int64 v17; // rdi
  int v18; // ebx
  unsigned int v19; // r15d
  unsigned int v20; // eax
  __int64 v21; // [rsp+0h] [rbp-40h]
  unsigned __int64 v22; // [rsp+8h] [rbp-38h]

  v2 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)a1;
  if ( v2 || (result = *(unsigned int *)(a1 + 20), (_DWORD)result) )
  {
    result = (unsigned int)(4 * v2);
    v4 = *(_QWORD ***)(a1 + 8);
    v5 = *(unsigned int *)(a1 + 24);
    if ( (unsigned int)result < 0x40 )
      result = 64;
    v21 = 32 * v5;
    v6 = &v4[4 * v5];
    if ( (unsigned int)v5 <= (unsigned int)result )
    {
      v7 = v4 + 1;
      if ( v4 != v6 )
      {
        while ( 1 )
        {
          v8 = (__int64)*(v7 - 1);
          if ( v8 != -4096 )
          {
            if ( v8 != -8192 )
            {
              v9 = *v7;
              while ( v7 != v9 )
              {
                v10 = (unsigned __int64)v9;
                v9 = (_QWORD *)*v9;
                v11 = *(_QWORD *)(v10 + 24);
                if ( v11 )
                  (*(void (__fastcall **)(__int64))(*(_QWORD *)v11 + 8LL))(v11);
                j_j___libc_free_0(v10);
              }
            }
            *(v7 - 1) = (_QWORD *)-4096LL;
          }
          result = (__int64)(v7 + 4);
          if ( v6 == v7 + 3 )
            break;
          v7 += 4;
        }
      }
      goto LABEL_20;
    }
    for ( i = v4 + 1; ; i += 4 )
    {
      v14 = (__int64)*(i - 1);
      if ( v14 != -4096 && v14 != -8192 )
      {
        v15 = *i;
        while ( v15 != i )
        {
          v16 = (unsigned __int64)v15;
          v15 = (_QWORD *)*v15;
          v17 = *(_QWORD *)(v16 + 24);
          if ( v17 )
          {
            v22 = v16;
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v17 + 8LL))(v17);
            v16 = v22;
          }
          j_j___libc_free_0(v16);
        }
      }
      if ( v6 == i + 3 )
        break;
    }
    v12 = *(_DWORD *)(a1 + 24);
    if ( v2 )
    {
      v18 = 64;
      v19 = v2 - 1;
      if ( v19 )
      {
        _BitScanReverse(&v20, v19);
        v18 = 1 << (33 - (v20 ^ 0x1F));
        if ( v18 < 64 )
          v18 = 64;
      }
      if ( v18 != v12 )
      {
        sub_C7D6A0(*(_QWORD *)(a1 + 8), v21, 8);
        result = sub_AF1560(4 * v18 / 3u + 1);
        *(_DWORD *)(a1 + 24) = result;
        if ( !(_DWORD)result )
          goto LABEL_19;
        *(_QWORD *)(a1 + 8) = sub_C7D670(32LL * (unsigned int)result, 8);
      }
    }
    else if ( v12 )
    {
      result = sub_C7D6A0(*(_QWORD *)(a1 + 8), v21, 8);
      *(_DWORD *)(a1 + 24) = 0;
LABEL_19:
      *(_QWORD *)(a1 + 8) = 0;
LABEL_20:
      *(_QWORD *)(a1 + 16) = 0;
      return result;
    }
    return (__int64)sub_227BE90(a1);
  }
  return result;
}
