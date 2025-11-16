// Function: sub_2B0CC60
// Address: 0x2b0cc60
//
bool __fastcall sub_2B0CC60(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 i; // rbx
  int v6; // r12d
  __int64 v7; // rax
  _QWORD *v8; // rsi
  __int64 v10[7]; // [rsp+8h] [rbp-38h] BYREF

  for ( i = a1; a2 != i; i = *(_QWORD *)(i + 8) )
  {
    v6 = *(_DWORD *)(a3 + 192);
    if ( v6 != (unsigned int)sub_BD2910(i) )
    {
      v10[0] = *(_QWORD *)(i + 24);
      v7 = *(_QWORD *)(a3 + 184);
      v8 = (_QWORD *)(*(_QWORD *)v7 + 8LL * *(unsigned int *)(v7 + 8));
      if ( v8 != sub_2B0B340(*(_QWORD **)v7, (__int64)v8, v10) )
        break;
    }
  }
  return a2 != i;
}
