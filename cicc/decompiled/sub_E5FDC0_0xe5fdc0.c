// Function: sub_E5FDC0
// Address: 0xe5fdc0
//
__int64 __fastcall sub_E5FDC0(__int64 a1, _QWORD *a2, unsigned int a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r14
  unsigned __int64 v8; // rax
  __int64 v9; // rcx
  __int64 v10; // r14
  __int64 v11; // rax
  __int64 v13; // r15
  __int64 v14; // rax
  __int64 v15; // rdx

  v6 = a3 - 1;
  v8 = *(unsigned int *)(a1 + 72);
  if ( (unsigned int)v6 < (unsigned int)v8 )
    goto LABEL_2;
  v13 = a3;
  if ( a3 == v8 )
    goto LABEL_2;
  if ( a3 < v8 )
  {
    *(_DWORD *)(a1 + 72) = a3;
LABEL_2:
    v9 = *(_QWORD *)(a1 + 64);
    goto LABEL_3;
  }
  if ( a3 > (unsigned __int64)*(unsigned int *)(a1 + 76) )
  {
    sub_C8D5F0(a1 + 64, (const void *)(a1 + 80), a3, 0x20u, a5, a6);
    v8 = *(unsigned int *)(a1 + 72);
  }
  v9 = *(_QWORD *)(a1 + 64);
  v14 = v9 + 32 * v8;
  v15 = v9 + 32 * v13;
  if ( v14 != v15 )
  {
    do
    {
      if ( v14 )
      {
        *(_DWORD *)v14 = 0;
        *(_BYTE *)(v14 + 4) = 0;
        *(_BYTE *)(v14 + 5) = 0;
        *(_QWORD *)(v14 + 8) = 0;
        *(_QWORD *)(v14 + 16) = 0;
        *(_QWORD *)(v14 + 24) = 0;
      }
      v14 += 32;
    }
    while ( v15 != v14 );
    v9 = *(_QWORD *)(a1 + 64);
  }
  *(_DWORD *)(a1 + 72) = a3;
LABEL_3:
  v10 = 32 * v6;
  if ( *(_BYTE *)(a1 + 304) )
    return sub_E9A500(a2, *(_QWORD *)(v9 + v10 + 24), 4, 0);
  v11 = sub_E808D0(*(_QWORD *)(v9 + v10 + 24), 0, a2[1], 0);
  return (*(__int64 (__fastcall **)(_QWORD *, __int64, __int64, _QWORD))(*a2 + 528LL))(a2, v11, 4, 0);
}
