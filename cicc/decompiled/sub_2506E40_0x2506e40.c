// Function: sub_2506E40
// Address: 0x2506e40
//
__int64 __fastcall sub_2506E40(__int64 *a1, __int64 a2)
{
  unsigned __int8 *v2; // r9
  int v3; // eax
  unsigned __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v7; // rbx
  __int64 v8; // rdx
  int v9; // eax
  __int64 v10; // rdi
  int v11; // esi
  unsigned int v12; // ecx
  __int64 *v13; // rax
  __int64 v14; // r8
  int v15; // eax
  int v16; // r9d

  v2 = *(unsigned __int8 **)(a2 + 24);
  v3 = *v2;
  if ( (unsigned __int8)v3 <= 0x1Cu )
    return 0;
  v5 = (unsigned int)(v3 - 34);
  if ( (unsigned __int8)v5 > 0x33u )
    return 0;
  v6 = 0x8000000000041LL;
  if ( !_bittest64(&v6, v5) )
    return 0;
  v7 = *a1;
  v8 = sub_B491C0((__int64)v2);
  v9 = *(_DWORD *)(v7 + 24);
  v10 = *(_QWORD *)(v7 + 8);
  if ( v9 )
  {
    v11 = v9 - 1;
    v12 = (v9 - 1) & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
    v13 = (__int64 *)(v10 + 16LL * v12);
    v14 = *v13;
    if ( v8 == *v13 )
    {
LABEL_7:
      LOBYTE(v14) = v13[1] == 0;
      return (unsigned int)v14;
    }
    v15 = 1;
    while ( v14 != -4096 )
    {
      v16 = v15 + 1;
      v12 = v11 & (v15 + v12);
      v13 = (__int64 *)(v10 + 16LL * v12);
      v14 = *v13;
      if ( v8 == *v13 )
        goto LABEL_7;
      v15 = v16;
    }
  }
  return 1;
}
