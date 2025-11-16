// Function: sub_3351CF0
// Address: 0x3351cf0
//
__int64 __fastcall sub_3351CF0(__int64 a1, __int64 a2)
{
  _QWORD *v2; // rbx
  unsigned __int64 v3; // rsi
  _QWORD *v5; // [rsp-70h] [rbp-70h]
  unsigned int v6; // [rsp-60h] [rbp-60h] BYREF
  int v7; // [rsp-5Ch] [rbp-5Ch] BYREF
  __int64 v8; // [rsp-58h] [rbp-58h] BYREF
  __int64 v9; // [rsp-50h] [rbp-50h]

  if ( !*(_QWORD *)(a1 + 80) )
    return 0;
  v2 = *(_QWORD **)(a2 + 40);
  v5 = &v2[2 * *(unsigned int *)(a2 + 48)];
  if ( v2 == v5 )
    return 0;
  while ( 1 )
  {
    if ( (*v2 & 6) == 0 )
    {
      v3 = *v2 & 0xFFFFFFFFFFFFFFF8LL;
      if ( *(_WORD *)(v3 + 250) )
      {
        sub_335E470(&v8, v3, *(_QWORD *)(a1 + 88));
        if ( v9 )
          break;
      }
    }
LABEL_4:
    v2 += 2;
    if ( v5 == v2 )
      return 0;
  }
  while ( 1 )
  {
    sub_3351990(
      (__int64)&v8,
      *(_QWORD *)(a1 + 80),
      *(_QWORD **)(a1 + 64),
      *(_QWORD *)(a1 + 72),
      &v6,
      &v7,
      *(_QWORD *)(a1 + 56));
    if ( (unsigned int)(*(_DWORD *)(*(_QWORD *)(a1 + 120) + 4LL * v6) + v7) >= *(_DWORD *)(*(_QWORD *)(a1 + 144)
                                                                                         + 4LL * v6) )
      return 1;
    sub_335E3B0(&v8);
    if ( !v9 )
      goto LABEL_4;
  }
}
