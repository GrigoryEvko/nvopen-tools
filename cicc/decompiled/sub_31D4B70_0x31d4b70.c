// Function: sub_31D4B70
// Address: 0x31d4b70
//
__int64 __fastcall sub_31D4B70(__int64 a1, _BYTE *a2, __int64 a3)
{
  __int64 result; // rax
  __int64 v5; // rax

  result = a2[32] & 0xF;
  if ( (unsigned __int8)result <= 5u )
  {
    if ( (unsigned __int8)result <= 1u )
    {
      if ( (_BYTE)result )
        goto LABEL_19;
      return (*(__int64 (__fastcall **)(_QWORD, __int64, __int64))(**(_QWORD **)(a1 + 224) + 296LL))(
               *(_QWORD *)(a1 + 224),
               a3,
               9);
    }
  }
  else
  {
    if ( (unsigned __int8)result <= 8u )
    {
      if ( (_BYTE)result != 6 )
        return result;
LABEL_19:
      BUG();
    }
    if ( (_BYTE)result != 10 )
      goto LABEL_19;
  }
  v5 = *(_QWORD *)(a1 + 208);
  if ( !*(_BYTE *)(v5 + 18) )
  {
    if ( !*(_BYTE *)(v5 + 313) || !sub_B326A0((__int64)a2) )
      return (*(__int64 (__fastcall **)(_QWORD, __int64, __int64))(**(_QWORD **)(a1 + 224) + 296LL))(
               *(_QWORD *)(a1 + 224),
               a3,
               24);
    return (*(__int64 (__fastcall **)(_QWORD, __int64, __int64))(**(_QWORD **)(a1 + 224) + 296LL))(
             *(_QWORD *)(a1 + 224),
             a3,
             9);
  }
  (*(void (__fastcall **)(_QWORD, __int64, __int64))(**(_QWORD **)(a1 + 224) + 296LL))(*(_QWORD *)(a1 + 224), a3, 9);
  if ( *(_BYTE *)(*(_QWORD *)(a1 + 208) + 312LL) && (unsigned __int8)sub_B2FE60(a2) )
    return (*(__int64 (__fastcall **)(_QWORD, __int64, __int64))(**(_QWORD **)(a1 + 224) + 296LL))(
             *(_QWORD *)(a1 + 224),
             a3,
             27);
  else
    return (*(__int64 (__fastcall **)(_QWORD, __int64, __int64))(**(_QWORD **)(a1 + 224) + 296LL))(
             *(_QWORD *)(a1 + 224),
             a3,
             25);
}
