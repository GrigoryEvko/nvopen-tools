// Function: sub_35E79A0
// Address: 0x35e79a0
//
void __fastcall sub_35E79A0(__int64 *a1, int *a2, __int64 a3)
{
  __int64 v3; // r13
  __int64 v6; // r14
  __int64 v7; // rax
  __int64 v8; // rdx
  int v9; // ecx
  int v10; // edi
  __int64 v11; // rsi
  __int64 v12; // rax
  int v13; // r8d
  __int64 v14; // r9
  int v15; // ecx
  __int64 v16; // rax

  v3 = 0x555555555555555LL;
  if ( a3 <= 0x555555555555555LL )
    v3 = a3;
  *a1 = a3;
  a1[1] = 0;
  a1[2] = 0;
  if ( a3 > 0 )
  {
    while ( 1 )
    {
      v6 = 24 * v3;
      v7 = sub_2207800(24 * v3);
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
    v12 = v7 + 24;
    v13 = a2[2];
    v14 = *((_QWORD *)a2 + 2);
    *(_DWORD *)(v12 - 24) = *a2;
    *(_DWORD *)(v12 - 20) = v10;
    *(_DWORD *)(v12 - 16) = v13;
    *(_QWORD *)(v12 - 8) = v14;
    if ( v11 != v12 )
    {
      do
      {
        v15 = *(_DWORD *)(v12 - 24);
        v12 += 24;
        *(_DWORD *)(v12 - 24) = v15;
        *(_DWORD *)(v12 - 20) = *(_DWORD *)(v12 - 44);
        *(_DWORD *)(v12 - 16) = *(_DWORD *)(v12 - 40);
        *(_QWORD *)(v12 - 8) = *(_QWORD *)(v12 - 32);
      }
      while ( v11 != v12 );
      v16 = v8 + v6 - 48;
      v14 = *(_QWORD *)(v16 + 40);
      v13 = *(_DWORD *)(v16 + 32);
      v10 = *(_DWORD *)(v16 + 28);
      v9 = *(_DWORD *)(v16 + 24);
    }
    *((_QWORD *)a2 + 2) = v14;
    a2[2] = v13;
    a2[1] = v10;
    *a2 = v9;
    a1[2] = v8;
    a1[1] = v3;
  }
}
