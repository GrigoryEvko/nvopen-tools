// Function: sub_BC8A80
// Address: 0xbc8a80
//
unsigned __int64 __fastcall sub_BC8A80(__int64 a1, __int64 a2)
{
  int v3; // r14d
  unsigned int v4; // eax
  unsigned int v5; // r8d
  unsigned int v6; // ebx
  unsigned __int64 result; // rax
  __int64 v8; // r15
  __int64 i; // rdx
  unsigned int j; // edx
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // rcx
  __int64 v14; // rdi
  unsigned __int8 v15; // al
  unsigned int v16; // [rsp+Ch] [rbp-34h]

  if ( (*(_BYTE *)(a1 - 16) & 2) != 0 )
    v3 = *(_DWORD *)(a1 - 24);
  else
    v3 = (*(_WORD *)(a1 - 16) >> 6) & 0xF;
  v4 = sub_BC8810(a1);
  v5 = v3 - v4;
  v6 = v4;
  result = *(unsigned int *)(a2 + 8);
  v8 = v5;
  if ( v5 != result )
  {
    if ( v5 >= result )
    {
      if ( v5 > (unsigned __int64)*(unsigned int *)(a2 + 12) )
      {
        v16 = v5;
        sub_C8D5F0(a2, a2 + 16, v5, 8);
        result = *(unsigned int *)(a2 + 8);
        v5 = v16;
      }
      result = *(_QWORD *)a2 + 8 * result;
      for ( i = *(_QWORD *)a2 + 8 * v8; i != result; result += 8LL )
      {
        if ( result )
          *(_QWORD *)result = 0;
      }
    }
    *(_DWORD *)(a2 + 8) = v5;
  }
  for ( j = 0; v3 != v6; *(_QWORD *)(*(_QWORD *)a2 + 8 * v14) = result )
  {
    v15 = *(_BYTE *)(a1 - 16);
    if ( (v15 & 2) != 0 )
      v11 = *(_QWORD *)(a1 - 32);
    else
      v11 = a1 + -16 - 8LL * ((v15 >> 2) & 0xF);
    v12 = *(_QWORD *)(v11 + 8LL * v6);
    if ( *(_BYTE *)v12 != 1 || (v13 = *(_QWORD *)(v12 + 136), *(_BYTE *)v13 != 17) )
      BUG();
    result = *(_QWORD *)(v13 + 24);
    if ( *(_DWORD *)(v13 + 32) > 0x40u )
      result = *(_QWORD *)result;
    v14 = j;
    ++v6;
    ++j;
  }
  return result;
}
