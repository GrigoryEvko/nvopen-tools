// Function: sub_38DCA60
// Address: 0x38dca60
//
void __fastcall sub_38DCA60(_QWORD *a1, __int64 a2)
{
  unsigned __int64 v2; // r12
  void (__fastcall *v3)(unsigned __int64); // rax

  v2 = *(_QWORD *)(a2 + 16);
  a1[1] = a2;
  *(_QWORD *)(a2 + 16) = a1;
  *a1 = &unk_4A3E610;
  if ( v2 )
  {
    v3 = *(void (__fastcall **)(unsigned __int64))(*(_QWORD *)v2 + 8LL);
    if ( v3 == sub_38DBD30 )
    {
      nullsub_1936();
      j_j___libc_free_0(v2);
    }
    else
    {
      v3(v2);
    }
  }
}
