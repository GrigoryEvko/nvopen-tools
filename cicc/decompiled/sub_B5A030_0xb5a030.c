// Function: sub_B5A030
// Address: 0xb5a030
//
void __fastcall sub_B5A030(__int64 a1, __int64 a2)
{
  _BYTE *v2; // rdi

  v2 = *(_BYTE **)(*(_QWORD *)(a1 + 32 * (2LL - (*(_DWORD *)(a1 + 4) & 0x7FFFFFF))) + 24LL);
  if ( v2 )
  {
    if ( *v2 )
      nullsub_2012();
    else
      sub_B91420(v2, a2);
  }
  else
  {
    nullsub_2011();
  }
}
