// Function: sub_2F683C0
// Address: 0x2f683c0
//
__int64 __fastcall sub_2F683C0(__int64 *a1, __int64 a2)
{
  __int64 v2; // rbx
  __int64 v3; // r14
  int v4; // r15d
  __int64 v5; // rax
  __int64 v6; // r8
  __int64 v7; // r9
  __int64 v8; // r13
  __int64 v9; // rax
  __int64 result; // rax
  __int64 v11; // r13
  __int64 *v12; // rdx

  v2 = *a1;
  if ( *(_DWORD *)(a2 + 8) )
  {
    v11 = *(_QWORD *)(v2 + 16);
    v12 = (__int64 *)sub_2E09D00((__int64 *)a2, v11);
    if ( v12 == (__int64 *)(*(_QWORD *)a2 + 24LL * *(unsigned int *)(a2 + 8))
      || (*(_DWORD *)((*v12 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v12 >> 1) & 3) > (*(_DWORD *)((v11 & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                                                                            | (unsigned int)(v11 >> 1)
                                                                                            & 3) )
    {
      v8 = 0;
    }
    else
    {
      v8 = v12[2];
    }
  }
  else
  {
    v3 = *(_QWORD *)(v2 + 16);
    v4 = *(_DWORD *)(a2 + 72);
    v5 = sub_A777F0(0x10u, *(__int64 **)v2);
    v8 = v5;
    if ( v5 )
    {
      *(_DWORD *)v5 = v4;
      *(_QWORD *)(v5 + 8) = v3;
    }
    v9 = *(unsigned int *)(a2 + 72);
    if ( v9 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 76) )
    {
      sub_C8D5F0(a2 + 64, (const void *)(a2 + 80), v9 + 1, 8u, v6, v7);
      v9 = *(unsigned int *)(a2 + 72);
    }
    *(_QWORD *)(*(_QWORD *)(a2 + 64) + 8 * v9) = v8;
    ++*(_DWORD *)(a2 + 72);
  }
  result = sub_2F60D70(a2, v8, *(_QWORD *)(v2 + 8), *(_QWORD *)(v2 + 24), v6, v7);
  **(_BYTE **)(v2 + 32) |= BYTE1(result);
  if ( (_BYTE)result )
  {
    result = *(_QWORD *)(*(_QWORD *)(v2 + 24) + 8LL);
    *(_QWORD *)(v8 + 8) = result;
  }
  return result;
}
