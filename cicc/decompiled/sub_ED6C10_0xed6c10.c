// Function: sub_ED6C10
// Address: 0xed6c10
//
__int64 __fastcall sub_ED6C10(__int64 a1)
{
  _BYTE *v1; // rdi
  __int64 (*v2)(void); // rax

  v1 = *(_BYTE **)(a1 + 128);
  v2 = *(__int64 (**)(void))(*(_QWORD *)v1 + 64LL);
  if ( (char *)v2 == (char *)sub_ED6030 )
    return v1[55] & 1;
  else
    return v2();
}
