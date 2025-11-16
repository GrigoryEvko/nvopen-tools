// Function: sub_D29010
// Address: 0xd29010
//
unsigned __int64 __fastcall sub_D29010(__int64 a1, __int64 a2)
{
  __int64 v2; // r14
  unsigned int v5; // esi
  __int64 v6; // rdi
  int v7; // r10d
  __int64 *v8; // r9
  unsigned int v9; // ecx
  _QWORD *v10; // rdx
  __int64 v11; // rax
  unsigned __int64 *v12; // rdx
  unsigned __int64 result; // rax
  int v14; // eax
  int v15; // edx
  __int64 v16; // rcx
  __int64 v17; // [rsp+0h] [rbp-30h] BYREF
  __int64 *v18; // [rsp+8h] [rbp-28h] BYREF

  v2 = a1 + 96;
  v17 = a2;
  v5 = *(_DWORD *)(a1 + 120);
  if ( !v5 )
  {
    ++*(_QWORD *)(a1 + 96);
    v18 = 0;
LABEL_20:
    v5 *= 2;
    goto LABEL_21;
  }
  v6 = *(_QWORD *)(a1 + 104);
  v7 = 1;
  v8 = 0;
  v9 = (v5 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v10 = (_QWORD *)(v6 + 16LL * v9);
  v11 = *v10;
  if ( a2 == *v10 )
  {
LABEL_3:
    v12 = v10 + 1;
    goto LABEL_4;
  }
  while ( v11 != -4096 )
  {
    if ( !v8 && v11 == -8192 )
      v8 = v10;
    v9 = (v5 - 1) & (v7 + v9);
    v10 = (_QWORD *)(v6 + 16LL * v9);
    v11 = *v10;
    if ( a2 == *v10 )
      goto LABEL_3;
    ++v7;
  }
  v14 = *(_DWORD *)(a1 + 112);
  if ( !v8 )
    v8 = v10;
  ++*(_QWORD *)(a1 + 96);
  v15 = v14 + 1;
  v18 = v8;
  if ( 4 * (v14 + 1) >= 3 * v5 )
    goto LABEL_20;
  v16 = a2;
  if ( v5 - *(_DWORD *)(a1 + 116) - v15 <= v5 >> 3 )
  {
LABEL_21:
    sub_D25040(v2, v5);
    sub_D24A00(v2, &v17, &v18);
    v16 = v17;
    v8 = v18;
    v15 = *(_DWORD *)(a1 + 112) + 1;
  }
  *(_DWORD *)(a1 + 112) = v15;
  if ( *v8 != -4096 )
    --*(_DWORD *)(a1 + 116);
  *v8 = v16;
  v12 = (unsigned __int64 *)(v8 + 1);
  v8[1] = 0;
LABEL_4:
  result = *v12;
  if ( !*v12 )
    return sub_D28F90((__int64 *)a1, a2, v12);
  return result;
}
