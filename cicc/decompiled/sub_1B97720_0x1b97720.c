// Function: sub_1B97720
// Address: 0x1b97720
//
__int64 __fastcall sub_1B97720(__int64 a1, int a2)
{
  __int64 v2; // r8
  int v3; // eax
  int v4; // esi
  int v5; // edx
  __int64 v7; // rdi
  int v8; // r8d
  unsigned int v9; // eax
  int v10; // ecx
  __int64 v11; // rax
  int v13; // [rsp+4h] [rbp-1Ch] BYREF
  __int64 v14; // [rsp+8h] [rbp-18h] BYREF

  v2 = 0;
  v3 = *(_DWORD *)(a1 + 40);
  v4 = *(_DWORD *)(a1 + 48) + a2;
  v13 = v4;
  if ( v3 )
  {
    v5 = v3 - 1;
    v7 = *(_QWORD *)(a1 + 24);
    v8 = 1;
    v9 = (v3 - 1) & (37 * v4);
    v10 = *(_DWORD *)(v7 + 16LL * v9);
    if ( v4 == v10 )
    {
LABEL_3:
      if ( (unsigned __int8)sub_1B97670(a1 + 16, &v13, &v14) )
        v11 = v14;
      else
        v11 = *(_QWORD *)(a1 + 24) + 16LL * *(unsigned int *)(a1 + 40);
      return *(_QWORD *)(v11 + 8);
    }
    else
    {
      while ( v10 != 0x7FFFFFFF )
      {
        v9 = v5 & (v8 + v9);
        v10 = *(_DWORD *)(v7 + 16LL * v9);
        if ( v4 == v10 )
          goto LABEL_3;
        ++v8;
      }
      return 0;
    }
  }
  return v2;
}
