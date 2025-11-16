// Function: sub_1B43970
// Address: 0x1b43970
//
__int16 __fastcall sub_1B43970(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  int v3; // r9d
  __int64 v4; // rdx
  __int64 v5; // r14
  __int64 v6; // r8
  __int64 v7; // r12
  __int64 v8; // rdx
  _QWORD *v9; // r13
  _QWORD *v10; // rdx
  __int64 v11; // rcx
  __int64 v13; // [rsp+0h] [rbp-40h]

  if ( !*(_QWORD *)(a1 + 48) && *(__int16 *)(a1 + 18) >= 0 )
    BUG();
  v2 = sub_1625790(a1, 2);
  v4 = *(unsigned int *)(v2 + 8);
  v5 = v2;
  if ( (unsigned int)v4 > 1 )
  {
    v2 = *(unsigned int *)(a2 + 8);
    v6 = v4;
    v7 = 1;
    while ( 1 )
    {
      v8 = *(_QWORD *)(*(_QWORD *)(v5 + 8 * (v7 - v4)) + 136LL);
      v9 = *(_QWORD **)(v8 + 24);
      if ( *(_DWORD *)(v8 + 32) > 0x40u )
        v9 = (_QWORD *)*v9;
      if ( *(_DWORD *)(a2 + 12) <= (unsigned int)v2 )
      {
        v13 = v6;
        sub_16CD150(a2, (const void *)(a2 + 16), 0, 8, v6, v3);
        v2 = *(unsigned int *)(a2 + 8);
        v6 = v13;
      }
      ++v7;
      *(_QWORD *)(*(_QWORD *)a2 + 8 * v2) = v9;
      v2 = (unsigned int)(*(_DWORD *)(a2 + 8) + 1);
      *(_DWORD *)(a2 + 8) = v2;
      if ( v6 == v7 )
        break;
      v4 = *(unsigned int *)(v5 + 8);
    }
  }
  if ( *(_BYTE *)(a1 + 16) == 26 )
  {
    LODWORD(v2) = *(unsigned __int16 *)(*(_QWORD *)(a1 - 72) + 18LL);
    BYTE1(v2) &= ~0x80u;
    if ( (_DWORD)v2 == 32 )
    {
      v2 = *(_QWORD *)a2;
      v10 = (_QWORD *)(*(_QWORD *)a2 + 8LL * *(unsigned int *)(a2 + 8) - 8);
      v11 = **(_QWORD **)a2;
      **(_QWORD **)a2 = *v10;
      *v10 = v11;
    }
  }
  return v2;
}
