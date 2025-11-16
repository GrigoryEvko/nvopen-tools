// Function: sub_D38870
// Address: 0xd38870
//
__int64 __fastcall sub_D38870(__int64 a1, __int64 a2)
{
  _QWORD *v2; // r12
  __int64 result; // rax
  __int64 v4; // r13
  __int64 v6; // rax
  _QWORD *v7; // r12
  _QWORD *v8; // rbx

  v2 = *(_QWORD **)a1;
  result = 9LL * *(unsigned int *)(a1 + 8);
  v4 = *(_QWORD *)a1 + 72LL * *(unsigned int *)(a1 + 8);
  if ( *(_QWORD *)a1 != v4 )
  {
    do
    {
      if ( a2 )
      {
        *(_QWORD *)a2 = 6;
        *(_QWORD *)(a2 + 8) = 0;
        v6 = v2[2];
        *(_QWORD *)(a2 + 16) = v6;
        if ( v6 != 0 && v6 != -4096 && v6 != -8192 )
          sub_BD6050((unsigned __int64 *)a2, *v2 & 0xFFFFFFFFFFFFFFF8LL);
        *(_QWORD *)(a2 + 24) = v2[3];
        *(_QWORD *)(a2 + 32) = v2[4];
        *(_BYTE *)(a2 + 40) = *((_BYTE *)v2 + 40);
        *(_DWORD *)(a2 + 44) = *((_DWORD *)v2 + 11);
        *(_DWORD *)(a2 + 48) = *((_DWORD *)v2 + 12);
        *(_QWORD *)(a2 + 56) = v2[7];
        *(_BYTE *)(a2 + 64) = *((_BYTE *)v2 + 64);
      }
      v2 += 9;
      a2 += 72;
    }
    while ( (_QWORD *)v4 != v2 );
    v7 = *(_QWORD **)a1;
    result = 9LL * *(unsigned int *)(a1 + 8);
    v8 = (_QWORD *)(*(_QWORD *)a1 + 72LL * *(unsigned int *)(a1 + 8));
    if ( *(_QWORD **)a1 != v8 )
    {
      do
      {
        result = *(v8 - 7);
        v8 -= 9;
        if ( result != -4096 && result != 0 && result != -8192 )
          result = sub_BD60C0(v8);
      }
      while ( v8 != v7 );
    }
  }
  return result;
}
