// Function: sub_2ECFA60
// Address: 0x2ecfa60
//
__int64 __fastcall sub_2ECFA60(__int64 a1, __int64 a2, unsigned int a3, unsigned int a4, __int64 a5, unsigned int a6)
{
  __int64 v8; // r15
  _DWORD *v10; // r15
  __int64 v11; // rax
  __int64 v12; // rdx
  unsigned int v13; // eax
  __int64 v15; // [rsp+8h] [rbp-48h]
  int v16; // [rsp+1Ch] [rbp-34h]

  v8 = 4LL * a3;
  v15 = a3;
  v16 = *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 8) + 208LL) + v8) * (a4 - a6);
  sub_2EC8EF0(a1, a3, v16);
  v10 = (_DWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 16) + 16LL) + v8);
  *v10 -= v16;
  v11 = *(unsigned int *)(a1 + 276);
  if ( (_DWORD)v11 != a3 )
  {
    v12 = *(_QWORD *)(a1 + 192);
    if ( (_DWORD)v11 )
      v13 = *(_DWORD *)(v12 + 4 * v11);
    else
      v13 = *(_DWORD *)(a1 + 184) * *(_DWORD *)(*(_QWORD *)(a1 + 8) + 288LL);
    if ( *(_DWORD *)(v12 + 4 * v15) > v13 )
      *(_DWORD *)(a1 + 276) = a3;
  }
  return sub_2ECE820((_QWORD *)a1, a2, a3, a4, a6);
}
