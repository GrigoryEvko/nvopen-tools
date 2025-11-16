// Function: sub_1DC6170
// Address: 0x1dc6170
//
void __fastcall sub_1DC6170(__int64 a1, __int64 a2, __int64 a3, int a4, unsigned int a5, int a6)
{
  __int64 v6; // r15
  __int64 i; // r13
  __int64 v8; // rbx
  __int64 v9; // r12

  v6 = a2;
  for ( i = *(_QWORD *)(a2 + 104); i; i = *(_QWORD *)(i + 104) )
  {
    v8 = *(_QWORD *)(i + 64);
    v9 = v8 + 8LL * *(unsigned int *)(i + 72);
    while ( v9 != v8 )
    {
      while ( 1 )
      {
        a2 = *(_QWORD *)(*(_QWORD *)v8 + 8LL);
        if ( (a2 & 0xFFFFFFFFFFFFFFF8LL) != 0 && (a2 & 6) != 0 )
          break;
        v8 += 8;
        if ( v9 == v8 )
          goto LABEL_8;
      }
      v8 += 8;
      sub_1DB79D0((__int64 *)v6, a2, *(__int64 **)(a1 + 32));
    }
LABEL_8:
    ;
  }
  sub_1DC3680((const __m128i *)a1, a2, a3, a4, a5, a6);
  sub_1DC5DD0(a1, v6, *(_DWORD *)(v6 + 112), -1, v6);
}
