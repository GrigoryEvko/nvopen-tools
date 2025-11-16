// Function: sub_1EE6140
// Address: 0x1ee6140
//
void __fastcall sub_1EE6140(__int64 a1)
{
  __int64 v2; // rax
  __int64 v3; // rax
  __int64 *v4; // rdi

  *(_QWORD *)(a1 + 40) = 0;
  v2 = *(_QWORD *)(a1 + 72);
  *(_QWORD *)(a1 + 32) = 0;
  if ( v2 != *(_QWORD *)(a1 + 80) )
    *(_QWORD *)(a1 + 80) = v2;
  v3 = *(_QWORD *)(a1 + 264);
  if ( v3 != *(_QWORD *)(a1 + 272) )
    *(_QWORD *)(a1 + 272) = v3;
  v4 = *(__int64 **)(a1 + 48);
  if ( *v4 != v4[1] )
  {
    v4[1] = *v4;
    v4 = *(__int64 **)(a1 + 48);
  }
  if ( *(_BYTE *)(a1 + 56) )
    sub_1EE5F10(v4);
  else
    sub_1EE5F50(v4);
  sub_1EE6130(a1 + 96);
  *(_DWORD *)(a1 + 208) = 0;
}
