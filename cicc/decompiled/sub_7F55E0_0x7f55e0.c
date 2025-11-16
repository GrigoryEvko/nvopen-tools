// Function: sub_7F55E0
// Address: 0x7f55e0
//
__int64 __fastcall sub_7F55E0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 result; // rax
  __int64 v4; // rax
  __int64 v5; // rax

  result = *(unsigned __int8 *)(a1 + 8);
  if ( (_BYTE)result == 2 )
  {
    *(_QWORD *)a3 = 0;
    *(_QWORD *)(a3 + 8) = 0;
    *(_QWORD *)(a3 + 24) = 0;
    *(_QWORD *)(a3 + 16) = 0;
    *(_QWORD *)(a3 + 32) = 0;
    *(_BYTE *)(a3 + 40) = 0;
    *(_QWORD *)a3 = *(_QWORD *)(a2 + 32);
    *(_QWORD *)(a2 + 32) = a3;
    v5 = *(_QWORD *)(a1 + 16);
    *(_QWORD *)(a3 + 24) = v5;
    result = *(_QWORD *)(v5 + 120);
    *(_QWORD *)(a3 + 8) = result;
  }
  else if ( (unsigned __int8)result > 2u )
  {
    if ( (_BYTE)result != 3 )
      sub_721090();
  }
  else
  {
    *(_QWORD *)a3 = 0;
    *(_QWORD *)(a3 + 8) = 0;
    *(_QWORD *)(a3 + 32) = 0;
    *(_QWORD *)(a3 + 16) = 0;
    *(_QWORD *)(a3 + 24) = 0;
    *(_BYTE *)(a3 + 40) = 0;
    *(_QWORD *)a3 = *(_QWORD *)(a2 + 32);
    *(_QWORD *)(a2 + 32) = a3;
    v4 = *(_QWORD *)(a1 + 16);
    *(_QWORD *)(a3 + 32) = v4;
    result = *(_QWORD *)(v4 + 40);
    *(_QWORD *)(a3 + 8) = result;
    *(_BYTE *)(a2 + 18) = 1;
  }
  return result;
}
