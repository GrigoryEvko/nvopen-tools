// Function: sub_6CC3B0
// Address: 0x6cc3b0
//
__int64 __fastcall sub_6CC3B0(__int64 a1, __int64 a2, _BYTE *a3)
{
  char v4; // al
  __int64 v5; // rdi
  __int64 result; // rax
  __int64 v7; // rbx

  if ( (unsigned int)sub_7307F0() )
  {
    v4 = *(_BYTE *)(a1 + 48);
    if ( v4 == 2 )
    {
      v5 = *(_QWORD *)(*(_QWORD *)(a1 + 56) + 144LL);
    }
    else if ( v4 == 8 )
    {
      sub_6E50A0(a1, a2);
      v5 = sub_7305B0(a1, a2);
    }
    else
    {
      v5 = sub_6E3F50(a1);
    }
    return sub_6F8800(v5, a2, a3);
  }
  else
  {
    v7 = *(_QWORD *)a2;
    *(_QWORD *)a2 = 0;
    *(_WORD *)(a2 + 8) = 183;
    result = sub_6CA0E0(a2, a1, 0, 0, 0, 0, a3, 0);
    *(_QWORD *)a2 = v7;
  }
  return result;
}
