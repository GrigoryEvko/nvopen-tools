// Function: sub_1315A80
// Address: 0x1315a80
//
__int64 __fastcall sub_1315A80(__int64 a1, __int64 a2, __int64 a3, unsigned __int64 a4)
{
  _QWORD *v6; // rdx
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // rax

  v6 = (_QWORD *)a4;
  if ( *(_DWORD *)(a2 + 78928) >= dword_5057900[0] )
  {
    if ( a3 == *(_QWORD *)(a4 + 216) )
    {
      v10 = *(_QWORD *)(a3 + 40);
      if ( a3 == v10 )
      {
        *(_QWORD *)(a4 + 216) = 0;
        return sub_13142A0(a2, a3, v6, a4);
      }
      *(_QWORD *)(a4 + 216) = v10;
    }
    *(_QWORD *)(*(_QWORD *)(a3 + 48) + 40LL) = *(_QWORD *)(*(_QWORD *)(a3 + 40) + 48LL);
    v8 = *(_QWORD *)(a3 + 48);
    *(_QWORD *)(*(_QWORD *)(a3 + 40) + 48LL) = v8;
    *(_QWORD *)(a3 + 48) = *(_QWORD *)(v8 + 40);
    v9 = *(_QWORD *)(a3 + 40);
    a4 = *(_QWORD *)(v9 + 48);
    *(_QWORD *)(a4 + 40) = v9;
    *(_QWORD *)(*(_QWORD *)(a3 + 48) + 40LL) = a3;
  }
  return sub_13142A0(a2, a3, v6, a4);
}
