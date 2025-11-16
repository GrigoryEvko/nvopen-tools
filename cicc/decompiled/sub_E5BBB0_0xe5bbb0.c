// Function: sub_E5BBB0
// Address: 0xe5bbb0
//
__int64 __fastcall sub_E5BBB0(__int64 a1, __int64 a2)
{
  __int64 v2; // r14
  _QWORD *v4; // rax
  _QWORD *v5; // rdx
  unsigned int v7; // r13d
  __int64 v8; // rdi
  __int64 *v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // r8
  __int64 v12; // r9
  __int64 *v13; // rax
  __int64 v14; // [rsp+0h] [rbp-40h] BYREF
  __int64 v15; // [rsp+8h] [rbp-38h]
  __int64 v16; // [rsp+10h] [rbp-30h]
  int v17; // [rsp+18h] [rbp-28h]

  v2 = a1 + 80;
  if ( *(_BYTE *)(a1 + 108) )
  {
    v4 = *(_QWORD **)(a1 + 88);
    v5 = &v4[*(unsigned int *)(a1 + 100)];
    if ( v4 == v5 )
      goto LABEL_8;
    while ( a2 != *v4 )
    {
      if ( v5 == ++v4 )
        goto LABEL_8;
    }
    return 1;
  }
  if ( sub_C8CA60(a1 + 80, a2) )
    return 1;
LABEL_8:
  if ( (*(_BYTE *)(a2 + 9) & 0x70) == 0x20
    && (*(_BYTE *)(a2 + 8) |= 8u,
        v8 = *(_QWORD *)(a2 + 24),
        v14 = 0,
        v15 = 0,
        v16 = 0,
        v17 = 0,
        (unsigned __int8)sub_E81950(v8, &v14, 0, 0))
    && !v15
    && !v17
    && v14
    && (*(_DWORD *)v14 & 0xFFFF00) == 0
    && (v7 = sub_E5BBB0(a1, *(_QWORD *)(v14 + 16)), (_BYTE)v7) )
  {
    if ( !*(_BYTE *)(a1 + 108) )
      goto LABEL_23;
    v13 = *(__int64 **)(a1 + 88);
    v10 = *(unsigned int *)(a1 + 100);
    v9 = &v13[v10];
    if ( v13 != v9 )
    {
      while ( a2 != *v13 )
      {
        if ( v9 == ++v13 )
          goto LABEL_21;
      }
      return v7;
    }
LABEL_21:
    if ( (unsigned int)v10 < *(_DWORD *)(a1 + 96) )
    {
      *(_DWORD *)(a1 + 100) = v10 + 1;
      *v9 = a2;
      ++*(_QWORD *)(a1 + 80);
    }
    else
    {
LABEL_23:
      sub_C8CC70(v2, a2, (__int64)v9, v10, v11, v12);
    }
  }
  else
  {
    return 0;
  }
  return v7;
}
