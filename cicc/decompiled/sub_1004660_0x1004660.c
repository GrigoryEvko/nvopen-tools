// Function: sub_1004660
// Address: 0x1004660
//
__int64 __fastcall sub_1004660(__int64 a1, __int64 a2, unsigned __int8 *a3, _DWORD *a4, __int64 a5)
{
  __int64 v8; // rax
  __int64 **v9; // rax
  __int64 v11; // [rsp+8h] [rbp-38h]

  v8 = *(_QWORD *)(a2 + 8);
  LODWORD(v11) = a5;
  BYTE4(v11) = *(_BYTE *)(v8 + 8) == 18;
  v9 = (__int64 **)sub_BCE1B0(*(__int64 **)(v8 + 24), v11);
  return sub_1004640(a2, a3, a4, a5, v9, a1 + 24);
}
