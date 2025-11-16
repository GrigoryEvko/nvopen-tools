// Function: sub_2E2E630
// Address: 0x2e2e630
//
__int64 __fastcall sub_2E2E630(__int64 a1, __int64 a2)
{
  __int64 v3; // rbx
  __int64 v4; // rax
  __int64 v5; // rdi
  __int64 (*v6)(); // rax
  unsigned int v7; // r15d
  __int64 v9; // r8
  __int64 v10; // r9
  unsigned __int64 v11; // rax
  int v12; // r14d
  __int64 v13; // rdx
  _QWORD *v14; // rax
  __int64 i; // rdx

  v3 = *(_QWORD *)(a2 + 48);
  v4 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a2 + 16) + 200LL))(*(_QWORD *)(a2 + 16));
  v5 = v4;
  if ( *(_DWORD *)(v3 + 32) == -858993459 * (unsigned int)((__int64)(*(_QWORD *)(v3 + 16) - *(_QWORD *)(v3 + 8)) >> 3) )
    return 0;
  v6 = *(__int64 (**)())(*(_QWORD *)v4 + 512LL);
  if ( v6 == sub_2E2CB30 )
    return 0;
  v7 = ((__int64 (__fastcall *)(__int64, __int64))v6)(v5, a2);
  if ( !(_BYTE)v7 )
  {
    return 0;
  }
  else
  {
    v11 = *(unsigned int *)(a1 + 8);
    v12 = -858993459 * ((__int64)(*(_QWORD *)(v3 + 16) - *(_QWORD *)(v3 + 8)) >> 3) - *(_DWORD *)(v3 + 32);
    v13 = v12;
    if ( v12 != v11 )
    {
      if ( v12 >= v11 )
      {
        if ( v12 > (unsigned __int64)*(unsigned int *)(a1 + 12) )
        {
          sub_C8D5F0(a1, (const void *)(a1 + 16), v12, 8u, v9, v10);
          v11 = *(unsigned int *)(a1 + 8);
          v13 = v12;
        }
        v14 = (_QWORD *)(*(_QWORD *)a1 + 8 * v11);
        for ( i = *(_QWORD *)a1 + 8 * v13; (_QWORD *)i != v14; ++v14 )
        {
          if ( v14 )
            *v14 = 0;
        }
      }
      *(_DWORD *)(a1 + 8) = v12;
    }
    sub_2E2E000((_QWORD *)a1, *(_QWORD *)(a2 + 16), *(_QWORD *)(a2 + 48));
    *(_BYTE *)(v3 + 665) = sub_2E2CDC0((_QWORD *)a1, a2);
  }
  return v7;
}
