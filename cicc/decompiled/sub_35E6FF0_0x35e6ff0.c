// Function: sub_35E6FF0
// Address: 0x35e6ff0
//
__int64 __fastcall sub_35E6FF0(__int64 a1)
{
  __int64 v1; // rax
  _BYTE *v2; // r12
  _BYTE *v3; // r13
  _BYTE *v4; // rax
  _BYTE *v5; // rbx
  __int64 *v6; // r15
  __int64 *i; // r13
  __int64 v8; // rbx
  __int64 v9; // r12
  __int64 v10; // rdx

  v1 = *(_QWORD *)(a1 + 24);
  v2 = *(_BYTE **)(v1 + 56);
  v3 = (_BYTE *)(v1 + 48);
  if ( v2 != (_BYTE *)(v1 + 48) )
  {
    while ( 1 )
    {
      if ( !v2 )
        BUG();
      v4 = v2;
      if ( (*v2 & 4) == 0 && (v2[44] & 8) != 0 )
      {
        do
          v4 = (_BYTE *)*((_QWORD *)v4 + 1);
        while ( (v4[44] & 8) != 0 );
      }
      v5 = (_BYTE *)*((_QWORD *)v4 + 1);
      sub_2FAD510(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 8) + 48LL) + 32LL), (__int64)v2);
      sub_2E88E20((__int64)v2);
      if ( v3 == v5 )
        break;
      v2 = v5;
    }
  }
  v6 = *(__int64 **)(a1 + 80);
  for ( i = &v6[*(unsigned int *)(a1 + 88)]; i != v6; *(_QWORD *)(v9 + 48) = *(_QWORD *)(v9 + 48) & 7LL | v8 )
  {
    v8 = *v6;
    v9 = *(_QWORD *)(a1 + 24);
    ++v6;
    sub_2E31040((__int64 *)(v9 + 40), v8);
    v10 = *(_QWORD *)(v9 + 48);
    *(_QWORD *)(v8 + 8) = v9 + 48;
    v10 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)v8 = v10 | *(_QWORD *)v8 & 7LL;
    *(_QWORD *)(v10 + 8) = v8;
  }
  return (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 128LL))(a1);
}
