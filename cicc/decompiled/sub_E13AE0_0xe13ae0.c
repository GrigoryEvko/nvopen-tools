// Function: sub_E13AE0
// Address: 0xe13ae0
//
__int64 __fastcall sub_E13AE0(__int64 a1, __int64 *a2)
{
  __int64 v3; // rdi
  char v4; // al
  __int64 result; // rax
  char v6; // al
  __int64 v7; // rax

  v3 = *(_QWORD *)(a1 + 16);
  if ( *(_BYTE *)(v3 + 8) != 11
    || (v7 = *(_QWORD *)(v3 + 16), *(_BYTE *)(v7 + 8) != 8)
    || *(_QWORD *)(v7 + 16) != 11
    || (result = *(_QWORD *)(v7 + 24), *(_QWORD *)result != 0x6A626F5F636A626FLL)
    || *(_WORD *)(result + 8) != 25445
    || *(_BYTE *)(result + 10) != 116 )
  {
    v4 = *(_BYTE *)(v3 + 10);
    if ( (v4 & 3) == 2 )
    {
      if ( (*(unsigned __int8 (__fastcall **)(__int64, __int64 *))(*(_QWORD *)v3 + 8LL))(v3, a2) )
        goto LABEL_4;
      v3 = *(_QWORD *)(a1 + 16);
      v4 = *(_BYTE *)(v3 + 10);
    }
    else if ( (v4 & 3) == 0 )
    {
      goto LABEL_4;
    }
    v6 = v4 & 0xC;
    if ( v6 == 8 )
    {
      if ( !(*(unsigned __int8 (__fastcall **)(__int64, __int64 *))(*(_QWORD *)v3 + 16LL))(v3, a2) )
        return (*(__int64 (__fastcall **)(_QWORD, __int64 *))(**(_QWORD **)(a1 + 16) + 40LL))(*(_QWORD *)(a1 + 16), a2);
    }
    else if ( v6 )
    {
      return (*(__int64 (__fastcall **)(__int64, __int64 *))(*(_QWORD *)v3 + 40LL))(v3, a2);
    }
LABEL_4:
    sub_E12F20(a2, 1u, ")");
    return (*(__int64 (__fastcall **)(_QWORD, __int64 *))(**(_QWORD **)(a1 + 16) + 40LL))(*(_QWORD *)(a1 + 16), a2);
  }
  return result;
}
