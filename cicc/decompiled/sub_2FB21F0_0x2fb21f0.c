// Function: sub_2FB21F0
// Address: 0x2fb21f0
//
__int64 __fastcall sub_2FB21F0(
        _QWORD *a1,
        __int64 a2,
        unsigned __int64 a3,
        __int64 a4,
        __int64 a5,
        unsigned __int64 a6)
{
  _QWORD *v6; // r12
  _QWORD *v7; // r13
  __int64 v8; // rbx
  __int64 *v9; // r14
  __int64 *v10; // rdx
  __int64 v11; // r8
  unsigned __int64 v12; // r9
  __int64 result; // rax
  __int64 v14; // r13
  __int64 v15; // r14
  __int64 v16; // rcx
  __int64 v17; // r8
  unsigned __int64 v18; // r9
  int v19; // esi
  __int64 v20; // r13
  __int64 v21; // r14
  unsigned __int16 v22; // dx
  _QWORD *v23; // rdx
  __int64 v24; // rdx
  unsigned __int64 v25; // [rsp+8h] [rbp-38h]

  v6 = (_QWORD *)a2;
  v7 = *(_QWORD **)(a2 + 104);
  if ( !v7 )
    return sub_2E0E3F0(a2, a3, a3, a4, a5, a6);
  v8 = *(_QWORD *)(a3 + 8);
  if ( (_BYTE)a4 )
  {
    v25 = v8 & 0xFFFFFFFFFFFFFFF8LL;
    do
    {
      while ( 1 )
      {
        v9 = sub_2FB21B0(v7[14], v7[15], *(_QWORD *)(a1[9] + 8LL));
        v10 = (__int64 *)sub_2E09D00(v9, v8);
        result = *v9 + 24LL * *((unsigned int *)v9 + 2);
        if ( v10 != (__int64 *)result )
        {
          result = *(_DWORD *)((*v10 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v10 >> 1) & 3;
          if ( (unsigned int)result <= ((unsigned int)(v8 >> 1) & 3 | *(_DWORD *)(v25 + 24)) )
          {
            result = v10[2];
            if ( result )
            {
              if ( *(_QWORD *)(result + 8) == v8 )
                break;
            }
          }
        }
        v7 = (_QWORD *)v7[13];
        if ( !v7 )
          return result;
      }
      result = sub_2E0E0B0((__int64)v7, v8, (__int64 *)(a1[1] + 56LL), v25, v11, v12);
      v7 = (_QWORD *)v7[13];
    }
    while ( v7 );
    return result;
  }
  v14 = *(_QWORD *)((v8 & 0xFFFFFFFFFFFFFFF8LL) + 16);
  v15 = *(_QWORD *)(v14 + 32);
  v16 = v15 + 40LL * (unsigned int)sub_2E88FE0(v14);
  result = *(_QWORD *)(v14 + 32);
  if ( v16 == result )
  {
    v20 = 0;
    v21 = 0;
    goto LABEL_19;
  }
  v19 = *(_DWORD *)(a2 + 112);
  v20 = 0;
  v21 = 0;
  while ( v19 != *(_DWORD *)(result + 8) )
  {
LABEL_13:
    result += 40;
    if ( result == v16 )
      goto LABEL_19;
  }
  v22 = (*(_DWORD *)result >> 8) & 0xFFF;
  if ( v22 )
  {
    v23 = (_QWORD *)(*(_QWORD *)(a1[6] + 272LL) + 16LL * v22);
    v21 |= *v23;
    v20 |= v23[1];
    goto LABEL_13;
  }
  result = sub_2EBF1E0(a1[3], v19);
  v21 = result;
  v20 = v24;
LABEL_19:
  while ( 1 )
  {
    v6 = (_QWORD *)v6[13];
    if ( !v6 )
      break;
    result = v20 & v6[15];
    if ( result | v21 & v6[14] )
      result = sub_2E0E0B0((__int64)v6, v8, (__int64 *)(a1[1] + 56LL), v16, v17, v18);
  }
  return result;
}
