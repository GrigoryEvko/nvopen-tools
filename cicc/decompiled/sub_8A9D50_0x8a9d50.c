// Function: sub_8A9D50
// Address: 0x8a9d50
//
__int64 __fastcall sub_8A9D50(__int64 a1, __int64 a2, int *a3)
{
  char v4; // al
  __int64 v5; // r15
  _QWORD *v7; // [rsp+8h] [rbp-58h]
  __int64 v8; // [rsp+18h] [rbp-48h] BYREF
  _QWORD **v9; // [rsp+20h] [rbp-40h] BYREF
  __int64 *v10; // [rsp+28h] [rbp-38h] BYREF

  v4 = *(_BYTE *)(a1 + 80);
  v8 = 0;
  v9 = 0;
  if ( (unsigned __int8)(v4 - 4) <= 1u )
  {
    v7 = (_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 88) + 168LL) + 176LL);
  }
  else
  {
    v7 = 0;
    if ( v4 == 7 )
      v7 = (_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 88) + 216LL) + 8LL);
  }
  v5 = *(_QWORD *)(a2 + 144);
  if ( v5 )
  {
    do
    {
      while ( 1 )
      {
        v10 = 0;
        if ( sub_8A64A0(v5, a1, 0, &v10) )
          break;
        v5 = *(_QWORD *)(v5 + 8);
        if ( !v5 )
          goto LABEL_9;
      }
      sub_8B5FF0(&v9, v5, v10);
      v5 = *(_QWORD *)(v5 + 8);
    }
    while ( v5 );
LABEL_9:
    if ( v9 )
      sub_893120(v9, a1, (__int64)&v8, v7, a3, a3 != 0);
    return v8;
  }
  return v5;
}
