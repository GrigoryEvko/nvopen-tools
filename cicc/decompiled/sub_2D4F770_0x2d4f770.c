// Function: sub_2D4F770
// Address: 0x2d4f770
//
__int64 __fastcall sub_2D4F770(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 (*v3)(); // rax
  __int64 v4; // r13
  __int64 (*v5)(); // rdx
  __int64 v6; // rax
  unsigned int v7; // r15d
  _QWORD *v8; // r14
  __int64 v9; // rsi
  unsigned int v10; // eax
  __int64 v12; // [rsp-50h] [rbp-50h]
  __int64 v13; // [rsp-40h] [rbp-40h]

  v3 = *(__int64 (**)())(*(_QWORD *)a3 + 16LL);
  if ( v3 == sub_23CE270 )
    BUG();
  v4 = ((__int64 (__fastcall *)(__int64))v3)(a3);
  if ( !(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v4 + 312LL))(v4) )
    return 0;
  v5 = *(__int64 (**)())(*(_QWORD *)v4 + 144LL);
  v6 = 0;
  if ( v5 != sub_2C8F680 )
    v6 = ((__int64 (__fastcall *)(__int64))v5)(v4);
  *(_QWORD *)a1 = v6;
  *(_QWORD *)(a1 + 8) = sub_B2BEC0(a2);
  v12 = a2 + 72;
  v13 = *(_QWORD *)(a2 + 80);
  if ( v13 == a2 + 72 )
  {
    return 0;
  }
  else
  {
    v7 = 0;
    do
    {
      if ( !v13 )
        BUG();
      v8 = (_QWORD *)(*(_QWORD *)(v13 + 24) & 0xFFFFFFFFFFFFFFF8LL);
      while ( (_QWORD *)(v13 + 24) != v8 )
      {
        v9 = (__int64)(v8 - 3);
        if ( !v8 )
          v9 = 0;
        v8 = (_QWORD *)(*v8 & 0xFFFFFFFFFFFFFFF8LL);
        v10 = sub_2D4C710((unsigned int *)a1, v9);
        if ( (_BYTE)v10 )
          v7 = v10;
      }
      v13 = *(_QWORD *)(v13 + 8);
    }
    while ( v12 != v13 );
  }
  return v7;
}
