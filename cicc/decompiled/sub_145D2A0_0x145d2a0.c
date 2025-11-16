// Function: sub_145D2A0
// Address: 0x145d2a0
//
__int64 __fastcall sub_145D2A0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // r13
  __int64 v6; // rdi
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // rax

  if ( *(_QWORD *)(a2 + 48) == a3 )
  {
    v5 = *(_QWORD *)a2;
    v6 = *(_QWORD *)(*(_QWORD *)a2 + 24LL);
    if ( *(_BYTE *)(a2 + 56) )
    {
      v10 = sub_15E0530(v6);
      v11 = sub_1643320(v10);
      v9 = sub_145CF80(v5, v11, 1, 0);
    }
    else
    {
      v7 = sub_15E0530(v6);
      v8 = sub_1643320(v7);
      v9 = sub_145CF80(v5, v8, 0, 0);
    }
    *(_QWORD *)a1 = v9;
    *(_BYTE *)(a1 + 8) = 1;
    return a1;
  }
  else
  {
    *(_BYTE *)(a1 + 8) = 0;
    return a1;
  }
}
