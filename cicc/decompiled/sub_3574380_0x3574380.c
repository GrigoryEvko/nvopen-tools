// Function: sub_3574380
// Address: 0x3574380
//
__int64 __fastcall sub_3574380(__int64 a1, int *a2)
{
  _QWORD *v3; // r13
  __int64 v4; // rax
  __int64 (*v5)(); // rdx
  int v7; // edx
  __int64 v8; // rax
  __int64 v9; // rdi
  int v10; // edx
  unsigned int v11; // ecx
  int v12; // esi
  int v13; // r8d
  __int64 v14; // rax
  unsigned __int64 i; // rbx
  __int64 v16; // r8
  __int64 v17; // r9
  __int64 v18; // rax

  v3 = *(_QWORD **)(*(_QWORD *)(a1 + 216) + 32LL);
  v4 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(*v3 + 16LL) + 200LL))(*(_QWORD *)(*v3 + 16LL));
  if ( (unsigned int)(*a2 - 1) <= 0x3FFFFFFE )
  {
    v5 = *(__int64 (**)())(*(_QWORD *)v4 + 168LL);
    if ( v5 != sub_2EA3FB0 )
    {
      if ( ((unsigned __int8 (__fastcall *)(__int64))v5)(v4) )
        return 257;
    }
  }
  if ( !sub_2DADE10((__int64)v3, *a2) )
    return 256;
  v7 = *(_DWORD *)(a1 + 264);
  v8 = (unsigned int)*a2;
  v9 = *(_QWORD *)(a1 + 248);
  if ( v7 )
  {
    v10 = v7 - 1;
    v11 = v10 & (37 * v8);
    v12 = *(_DWORD *)(v9 + 4LL * v11);
    if ( v12 == (_DWORD)v8 )
      return 256;
    v13 = 1;
    while ( v12 != -1 )
    {
      v11 = v10 & (v13 + v11);
      v12 = *(_DWORD *)(v9 + 4LL * v11);
      if ( v12 == (_DWORD)v8 )
        return 256;
      ++v13;
    }
  }
  if ( (int)v8 < 0 )
    v14 = *(_QWORD *)(v3[7] + 16 * (v8 & 0x7FFFFFFF) + 8);
  else
    v14 = *(_QWORD *)(v3[38] + 8 * v8);
  if ( v14 )
  {
    if ( (*(_BYTE *)(v14 + 3) & 0x10) == 0 )
    {
      v14 = *(_QWORD *)(v14 + 32);
      if ( v14 )
      {
        if ( (*(_BYTE *)(v14 + 3) & 0x10) == 0 )
          BUG();
      }
    }
  }
  for ( i = *(_QWORD *)(v14 + 16); (*(_BYTE *)(i + 44) & 4) != 0; i = *(_QWORD *)i & 0xFFFFFFFFFFFFFFF8LL )
    ;
  if ( !sub_2E5E6D0(*(_QWORD *)(a1 + 224), *(_QWORD *)(i + 24)) )
    return 257;
  v18 = *(unsigned int *)(a1 + 1584);
  if ( v18 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 1588) )
  {
    sub_C8D5F0(a1 + 1576, (const void *)(a1 + 1592), v18 + 1, 8u, v16, v17);
    v18 = *(unsigned int *)(a1 + 1584);
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 1576) + 8 * v18) = i;
  ++*(_DWORD *)(a1 + 1584);
  return 0;
}
