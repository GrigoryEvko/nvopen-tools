// Function: sub_2351F40
// Address: 0x2351f40
//
void __fastcall sub_2351F40(_BYTE *a1, __int64 a2)
{
  char v2; // al
  __int64 v3; // r8

  v2 = a1[24];
  if ( (v2 & 2) != 0 )
    sub_2351ED0(a1, a2);
  v3 = *(_QWORD *)a1;
  if ( (v2 & 1) != 0 )
  {
    if ( v3 )
      (*(void (__fastcall **)(_QWORD))(*(_QWORD *)v3 + 8LL))(*(_QWORD *)a1);
  }
  else if ( v3 )
  {
    j_j___libc_free_0(*(_QWORD *)a1);
  }
}
