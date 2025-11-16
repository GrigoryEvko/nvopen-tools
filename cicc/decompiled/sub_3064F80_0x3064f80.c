// Function: sub_3064F80
// Address: 0x3064f80
//
unsigned __int64 __fastcall sub_3064F80(__int64 a1, __int64 a2, __int64 *a3, char a4, char a5)
{
  unsigned __int64 v5; // rbx
  unsigned int i; // r12d
  __int64 v8; // rax
  __int64 *v9; // r11
  __int64 v10; // rax
  __int64 v11; // rdx
  unsigned int v12; // eax
  bool v13; // of
  __int64 *v14; // r11
  __int64 v15; // rax
  __int64 v16; // rdx
  unsigned int v17; // eax
  __int64 *v19; // [rsp+0h] [rbp-50h]
  __int64 *v20; // [rsp+0h] [rbp-50h]
  int v23; // [rsp+Ch] [rbp-44h]
  unsigned int v24; // [rsp+1Ch] [rbp-34h]

  if ( *(_BYTE *)(a2 + 8) == 18 )
    return 0;
  v23 = *(_DWORD *)(a2 + 32);
  if ( v23 <= 0 )
    return 0;
  v5 = 0;
  for ( i = 0; i != v23; ++i )
  {
    v8 = *a3;
    if ( *((_DWORD *)a3 + 2) > 0x40u )
      v8 = *(_QWORD *)(v8 + 8LL * (i >> 6));
    if ( (v8 & (1LL << i)) != 0 )
    {
      if ( a4 )
      {
        v9 = (__int64 *)a2;
        if ( (unsigned int)*(unsigned __int8 *)(a2 + 8) - 17 <= 1 )
          v9 = **(__int64 ***)(a2 + 16);
        v19 = v9;
        v10 = sub_2D5BAE0(*(_QWORD *)(a1 + 24), *(_QWORD *)(a1 + 8), v9, 0);
        BYTE2(v24) = 0;
        v12 = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, __int64, _QWORD))(**(_QWORD **)(a1 + 24) + 736LL))(
                *(_QWORD *)(a1 + 24),
                *v19,
                v10,
                v11,
                v24);
        v13 = __OFADD__(v12, v5);
        v5 += v12;
        if ( v13 )
        {
          v5 = 0x8000000000000000LL;
          if ( v12 )
            v5 = 0x7FFFFFFFFFFFFFFFLL;
        }
      }
      if ( a5 )
      {
        v14 = (__int64 *)a2;
        if ( (unsigned int)*(unsigned __int8 *)(a2 + 8) - 17 <= 1 )
          v14 = **(__int64 ***)(a2 + 16);
        v20 = v14;
        v15 = sub_2D5BAE0(*(_QWORD *)(a1 + 24), *(_QWORD *)(a1 + 8), v14, 0);
        BYTE2(v24) = 0;
        v17 = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, __int64, _QWORD))(**(_QWORD **)(a1 + 24) + 736LL))(
                *(_QWORD *)(a1 + 24),
                *v20,
                v15,
                v16,
                v24);
        v13 = __OFADD__(v17, v5);
        v5 += v17;
        if ( v13 )
        {
          v5 = 0x8000000000000000LL;
          if ( v17 )
            v5 = 0x7FFFFFFFFFFFFFFFLL;
        }
      }
    }
  }
  return v5;
}
