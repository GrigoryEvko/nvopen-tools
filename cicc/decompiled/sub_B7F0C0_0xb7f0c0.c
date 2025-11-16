// Function: sub_B7F0C0
// Address: 0xb7f0c0
//
__int64 __fastcall sub_B7F0C0(__int64 *a1, __int64 a2)
{
  int v2; // eax
  __int64 v3; // r15
  int v4; // r14d
  const void *v5; // r13
  size_t v6; // rdx
  size_t v7; // rbx
  unsigned int v8; // eax
  unsigned int v9; // r8d
  __int64 *v10; // rcx
  __int64 result; // rax
  __int64 v12; // rax
  unsigned int v13; // r8d
  __int64 *v14; // rcx
  __int64 v15; // r12
  __int64 *v16; // [rsp+0h] [rbp-40h]
  unsigned int v17; // [rsp+Ch] [rbp-34h]

  v2 = sub_B2BED0(a2);
  v3 = *a1;
  v4 = v2;
  v5 = (const void *)sub_BD5D20(a2);
  v7 = v6;
  v8 = sub_C92610(v5, v6);
  v9 = sub_C92740(v3, v5, v7, v8);
  v10 = (__int64 *)(*(_QWORD *)v3 + 8LL * v9);
  result = *v10;
  if ( *v10 )
  {
    if ( result != -8 )
    {
      *(_DWORD *)(result + 12) = v4;
      return result;
    }
    --*(_DWORD *)(v3 + 16);
  }
  v16 = v10;
  v17 = v9;
  v12 = sub_C7D670(v7 + 17, 8);
  v13 = v17;
  v14 = v16;
  v15 = v12;
  if ( v7 )
  {
    memcpy((void *)(v12 + 16), v5, v7);
    v14 = v16;
    v13 = v17;
  }
  *(_BYTE *)(v15 + v7 + 16) = 0;
  *(_QWORD *)v15 = v7;
  *(_DWORD *)(v15 + 8) = 0;
  *(_DWORD *)(v15 + 12) = v4;
  *v14 = v15;
  ++*(_DWORD *)(v3 + 12);
  return sub_C929D0(v3, v13);
}
