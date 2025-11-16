// Function: sub_28FFEE0
// Address: 0x28ffee0
//
void __fastcall sub_28FFEE0(__int64 a1, __int64 a2)
{
  _QWORD *v2; // r12
  unsigned __int64 v3; // r14
  __int64 v5; // rax
  __int64 v6; // rax
  int v7; // eax
  __int64 v8; // rax
  _QWORD *v9; // r12
  _QWORD *v10; // rbx
  __int64 v11; // rax
  __int64 v12; // rax

  v2 = *(_QWORD **)a1;
  v3 = *(_QWORD *)a1 + ((unsigned __int64)*(unsigned int *)(a1 + 8) << 6);
  if ( *(_QWORD *)a1 != v3 )
  {
    do
    {
      if ( a2 )
      {
        v5 = *v2;
        *(_QWORD *)(a2 + 8) = 0;
        *(_QWORD *)(a2 + 16) = 0;
        *(_QWORD *)a2 = v5;
        v6 = v2[3];
        *(_QWORD *)(a2 + 24) = v6;
        if ( v6 != 0 && v6 != -4096 && v6 != -8192 )
          sub_BD6050((unsigned __int64 *)(a2 + 8), v2[1] & 0xFFFFFFFFFFFFFFF8LL);
        v7 = *((_DWORD *)v2 + 8);
        *(_QWORD *)(a2 + 40) = 0;
        *(_QWORD *)(a2 + 48) = 0;
        *(_DWORD *)(a2 + 32) = v7;
        v8 = v2[7];
        *(_QWORD *)(a2 + 56) = v8;
        if ( v8 != -4096 && v8 != 0 && v8 != -8192 )
          sub_BD6050((unsigned __int64 *)(a2 + 40), v2[5] & 0xFFFFFFFFFFFFFFF8LL);
      }
      v2 += 8;
      a2 += 64;
    }
    while ( (_QWORD *)v3 != v2 );
    v9 = *(_QWORD **)a1;
    v10 = (_QWORD *)(*(_QWORD *)a1 + ((unsigned __int64)*(unsigned int *)(a1 + 8) << 6));
    if ( *(_QWORD **)a1 != v10 )
    {
      do
      {
        v11 = *(v10 - 1);
        v10 -= 8;
        if ( v11 != -4096 && v11 != 0 && v11 != -8192 )
          sub_BD60C0(v10 + 5);
        v12 = v10[3];
        if ( v12 != 0 && v12 != -4096 && v12 != -8192 )
          sub_BD60C0(v10 + 1);
      }
      while ( v10 != v9 );
    }
  }
}
