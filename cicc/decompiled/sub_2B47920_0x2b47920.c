// Function: sub_2B47920
// Address: 0x2b47920
//
void __fastcall sub_2B47920(__int64 *a1, int *a2, __int64 a3)
{
  __int64 v3; // r13
  __int64 v6; // r14
  __int64 v7; // rax
  __int64 v8; // rdx
  int v9; // ecx
  int v10; // edi
  __int64 v11; // rsi
  __int64 v12; // rax
  __int64 v13; // r8
  int v14; // ecx
  __int64 v15; // rax

  v3 = 0x7FFFFFFFFFFFFFFLL;
  if ( a3 <= 0x7FFFFFFFFFFFFFFLL )
    v3 = a3;
  *a1 = a3;
  a1[1] = 0;
  a1[2] = 0;
  if ( a3 > 0 )
  {
    while ( 1 )
    {
      v6 = 16 * v3;
      v7 = sub_2207800(16 * v3);
      v8 = v7;
      if ( v7 )
        break;
      v3 >>= 1;
      if ( !v3 )
        return;
    }
    v9 = *a2;
    v10 = a2[1];
    v11 = v7 + v6;
    v12 = v7 + 16;
    v13 = *((_QWORD *)a2 + 1);
    *(_DWORD *)(v12 - 16) = *a2;
    *(_DWORD *)(v12 - 12) = v10;
    *(_QWORD *)(v12 - 8) = v13;
    if ( v11 != v12 )
    {
      do
      {
        v14 = *(_DWORD *)(v12 - 16);
        v12 += 16;
        *(_DWORD *)(v12 - 16) = v14;
        *(_DWORD *)(v12 - 12) = *(_DWORD *)(v12 - 28);
        *(_QWORD *)(v12 - 8) = *(_QWORD *)(v12 - 24);
      }
      while ( v11 != v12 );
      v15 = v8 + v6 - 32;
      v13 = *(_QWORD *)(v15 + 24);
      v10 = *(_DWORD *)(v15 + 20);
      v9 = *(_DWORD *)(v15 + 16);
    }
    *((_QWORD *)a2 + 1) = v13;
    a2[1] = v10;
    *a2 = v9;
    a1[2] = v8;
    a1[1] = v3;
  }
}
