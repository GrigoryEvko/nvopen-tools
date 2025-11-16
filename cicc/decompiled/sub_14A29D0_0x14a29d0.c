// Function: sub_14A29D0
// Address: 0x14a29d0
//
char __fastcall sub_14A29D0(__int64 *a1, _BYTE *a2)
{
  __int64 v2; // rdi
  __int64 (*v3)(void); // rax

  v2 = *a1;
  v3 = *(__int64 (**)(void))(*(_QWORD *)v2 + 144LL);
  if ( (char *)v3 == (char *)sub_14A2460 )
    return sub_14A2090(v2 + 8, a2);
  else
    return v3();
}
