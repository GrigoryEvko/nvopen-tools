// Function: sub_2185E80
// Address: 0x2185e80
//
__int64 __fastcall sub_2185E80(__int64 a1, unsigned int a2)
{
  __int64 v2; // rax
  unsigned int v3; // edx
  __int64 v5; // rdi
  int *v6; // rcx
  int v7; // r8d
  char v8; // al
  _DWORD *v9; // rdx
  unsigned int v11; // esi
  int v12; // eax
  int v13; // eax
  int v14; // ecx
  int v15; // r10d
  unsigned int v16; // [rsp-34h] [rbp-34h] BYREF
  _QWORD v17[6]; // [rsp-30h] [rbp-30h] BYREF

  v2 = *(unsigned int *)(a1 + 64);
  if ( (_DWORD)v2 )
  {
    v3 = (v2 - 1) & (37 * a2);
    v5 = *(_QWORD *)(a1 + 48);
    v6 = (int *)(v5 + 8LL * v3);
    v7 = *v6;
    if ( a2 != *v6 )
    {
      v14 = 1;
      while ( v7 != 0x7FFFFFFF )
      {
        v15 = v14 + 1;
        v3 = (v2 - 1) & (v14 + v3);
        v6 = (int *)(v5 + 8LL * v3);
        v7 = *v6;
        if ( a2 == *v6 )
          goto LABEL_3;
        v14 = v15;
      }
      return 0;
    }
LABEL_3:
    if ( v6 == (int *)(v5 + 8 * v2) )
      return 0;
    v16 = a2;
    v8 = sub_1BFD870(a1 + 40, (int *)&v16, v17);
    v9 = (_DWORD *)v17[0];
    if ( v8 )
      return *(unsigned int *)(v17[0] + 4LL);
    v11 = *(_DWORD *)(a1 + 64);
    v12 = *(_DWORD *)(a1 + 56);
    ++*(_QWORD *)(a1 + 40);
    v13 = v12 + 1;
    if ( 4 * v13 >= 3 * v11 )
    {
      v11 *= 2;
    }
    else if ( v11 - *(_DWORD *)(a1 + 60) - v13 > v11 >> 3 )
    {
LABEL_8:
      *(_DWORD *)(a1 + 56) = v13;
      if ( *v9 != 0x7FFFFFFF )
        --*(_DWORD *)(a1 + 60);
      *(_QWORD *)v9 = v16;
      return 0;
    }
    sub_1C01850(a1 + 40, v11);
    sub_1BFD870(a1 + 40, (int *)&v16, v17);
    v9 = (_DWORD *)v17[0];
    v13 = *(_DWORD *)(a1 + 56) + 1;
    goto LABEL_8;
  }
  return 0;
}
