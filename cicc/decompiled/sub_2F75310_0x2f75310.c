// Function: sub_2F75310
// Address: 0x2f75310
//
void __fastcall sub_2F75310(__int64 a1)
{
  __int64 v2; // rax
  __int64 v3; // rax
  __int64 *v4; // rdi

  *(_QWORD *)(a1 + 40) = 0;
  v2 = *(_QWORD *)(a1 + 72);
  *(_QWORD *)(a1 + 32) = 0;
  if ( v2 != *(_QWORD *)(a1 + 80) )
    *(_QWORD *)(a1 + 80) = v2;
  v3 = *(_QWORD *)(a1 + 392);
  if ( v3 != *(_QWORD *)(a1 + 400) )
    *(_QWORD *)(a1 + 400) = v3;
  v4 = *(__int64 **)(a1 + 48);
  if ( *v4 != v4[1] )
  {
    v4[1] = *v4;
    v4 = *(__int64 **)(a1 + 48);
  }
  if ( *(_BYTE *)(a1 + 56) )
    sub_2F750F0(v4);
  else
    sub_2F75130(v4);
  sub_2F75300(a1 + 96);
  *(_DWORD *)(a1 + 336) = 0;
}
