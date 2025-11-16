// Function: sub_829EB0
// Address: 0x829eb0
//
__int64 __fastcall sub_829EB0(__int64 *a1, __int64 a2, _QWORD *a3, __int64 a4, __int64 a5)
{
  __int64 v8; // r15
  __int64 result; // rax
  __int64 v10; // rcx
  __int64 v11; // r8
  __int64 i; // rax
  _QWORD *v13; // r14
  char v14; // cl
  __int64 j; // rdi
  __m128i *v16; // rax
  __int64 v17; // rax

  v8 = *a1;
  if ( a5 )
  {
    result = sub_828890(a1, a5);
LABEL_3:
    if ( v8 != *a1 )
    {
      result = sub_8D97D0(*a1, v8, 0, v10, v11);
      if ( !(_DWORD)result )
      {
        result = sub_829DD0(*a1, a2, a3, a4, a5);
        if ( (_DWORD)result )
          *a1 = v8;
      }
    }
    return result;
  }
  for ( i = a4; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  result = *(_QWORD *)(*(_QWORD *)i + 96LL);
  v13 = *(_QWORD **)(result + 40);
  if ( v13 )
  {
    result = v13[1];
    if ( result )
    {
      do
      {
        v14 = *(_BYTE *)(result + 80);
        if ( v14 == 16 )
        {
          result = **(_QWORD **)(result + 88);
          v14 = *(_BYTE *)(result + 80);
        }
        if ( v14 == 24 )
          result = *(_QWORD *)(result + 88);
        for ( j = *(_QWORD *)(*(_QWORD *)(result + 88) + 152LL); *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
          ;
        v16 = sub_73D790(j);
        v17 = sub_6EEB30((__int64)v16, 0);
        result = sub_828890(a1, v17);
        v13 = (_QWORD *)*v13;
        if ( !v13 )
          break;
        result = v13[1];
      }
      while ( result );
      goto LABEL_3;
    }
  }
  return result;
}
