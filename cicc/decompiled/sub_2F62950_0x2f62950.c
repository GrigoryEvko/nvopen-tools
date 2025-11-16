// Function: sub_2F62950
// Address: 0x2f62950
//
__int64 __fastcall sub_2F62950(__int64 *a1, __int64 a2, _BYTE *a3)
{
  __int64 result; // rax
  __int64 v4; // r15
  __int64 v6; // rbx
  __int64 v7; // r12
  unsigned __int64 v8; // r14
  __int64 v9; // r15
  __int64 *v10; // rcx
  __int64 v11; // r8
  unsigned int v12; // r9d
  unsigned int v13; // edi
  __int64 v14; // rax
  __int64 v16; // [rsp+18h] [rbp-38h]

  result = *a1;
  v4 = *(unsigned int *)(*a1 + 72);
  v16 = 8 * v4;
  if ( (_DWORD)v4 )
  {
    v6 = 0;
    while ( 1 )
    {
      result = a1[16] + 8 * v6;
      if ( *(_DWORD *)result )
        goto LABEL_19;
      v7 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(*a1 + 64) + v6) + 8LL);
      v8 = v7 & 0xFFFFFFFFFFFFFFF8LL;
      if ( (v7 & 0xFFFFFFFFFFFFFFF8LL) == 0 || (v7 & 6) == 0 )
        goto LABEL_19;
      v9 = *(_QWORD *)(a2 + 104);
      if ( v9 )
        break;
LABEL_18:
      *(_BYTE *)(result + 57) = 1;
      result = (__int64)a3;
      *a3 = 1;
LABEL_19:
      v6 += 8;
      if ( v6 == v16 )
        return result;
    }
    while ( 1 )
    {
      v10 = (__int64 *)sub_2E09D00((__int64 *)v9, v8);
      v11 = *(_QWORD *)v9 + 24LL * *(unsigned int *)(v9 + 8);
      if ( v10 != (__int64 *)v11 )
      {
        v12 = *(_DWORD *)(v8 + 24);
        v13 = *(_DWORD *)((*v10 & 0xFFFFFFFFFFFFFFF8LL) + 24);
        if ( (unsigned __int64)(v13 | (*v10 >> 1) & 3) > v12 || v8 != (v10[1] & 0xFFFFFFFFFFFFFFF8LL) )
          goto LABEL_8;
        if ( (__int64 *)v11 != v10 + 3 )
          break;
      }
LABEL_11:
      v9 = *(_QWORD *)(v9 + 104);
      if ( !v9 )
      {
        result = a1[16] + 8 * v6;
        goto LABEL_18;
      }
    }
    v14 = v10[3];
    v10 += 3;
    v13 = *(_DWORD *)((v14 & 0xFFFFFFFFFFFFFFF8LL) + 24);
LABEL_8:
    if ( v12 >= v13 )
    {
      result = v10[2];
      if ( result )
      {
        if ( v7 == *(_QWORD *)(result + 8) )
          goto LABEL_19;
      }
    }
    goto LABEL_11;
  }
  return result;
}
