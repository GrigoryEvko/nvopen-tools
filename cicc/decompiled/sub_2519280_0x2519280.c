// Function: sub_2519280
// Address: 0x2519280
//
__int64 __fastcall sub_2519280(__int64 a1, __int64 *a2)
{
  unsigned int v4; // esi
  __int64 v5; // r9
  __int64 v6; // r8
  __int64 *v7; // r10
  int v8; // r11d
  __int64 result; // rax
  __int64 *v10; // rdi
  __int64 v11; // rcx
  int v12; // eax
  int v13; // edx
  __int64 v14; // r12
  __int64 *v15; // [rsp+8h] [rbp-28h] BYREF

  v4 = *(_DWORD *)(a1 + 24);
  if ( !v4 )
  {
    ++*(_QWORD *)a1;
    v15 = 0;
LABEL_19:
    v4 *= 2;
    goto LABEL_20;
  }
  v5 = v4 - 1;
  v6 = *(_QWORD *)(a1 + 8);
  v7 = 0;
  v8 = 1;
  result = (unsigned int)v5 & (((unsigned int)*a2 >> 9) ^ ((unsigned int)*a2 >> 4));
  v10 = (__int64 *)(v6 + 8 * result);
  v11 = *v10;
  if ( *a2 == *v10 )
    return result;
  while ( v11 != -4096 )
  {
    if ( v11 != -8192 || v7 )
      v10 = v7;
    result = (unsigned int)v5 & (v8 + (_DWORD)result);
    v11 = *(_QWORD *)(v6 + 8LL * (unsigned int)result);
    if ( *a2 == v11 )
      return result;
    ++v8;
    v7 = v10;
    v10 = (__int64 *)(v6 + 8LL * (unsigned int)result);
  }
  v12 = *(_DWORD *)(a1 + 16);
  if ( !v7 )
    v7 = v10;
  ++*(_QWORD *)a1;
  v13 = v12 + 1;
  v15 = v7;
  if ( 4 * (v12 + 1) >= 3 * v4 )
    goto LABEL_19;
  if ( v4 - *(_DWORD *)(a1 + 20) - v13 <= v4 >> 3 )
  {
LABEL_20:
    sub_A35F10(a1, v4);
    sub_A2AFD0(a1, a2, &v15);
    v7 = v15;
    v13 = *(_DWORD *)(a1 + 16) + 1;
  }
  *(_DWORD *)(a1 + 16) = v13;
  if ( *v7 != -4096 )
    --*(_DWORD *)(a1 + 20);
  *v7 = *a2;
  result = *(unsigned int *)(a1 + 40);
  v14 = *a2;
  if ( result + 1 > (unsigned __int64)*(unsigned int *)(a1 + 44) )
  {
    sub_C8D5F0(a1 + 32, (const void *)(a1 + 48), result + 1, 8u, v6, v5);
    result = *(unsigned int *)(a1 + 40);
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 32) + 8 * result) = v14;
  ++*(_DWORD *)(a1 + 40);
  return result;
}
