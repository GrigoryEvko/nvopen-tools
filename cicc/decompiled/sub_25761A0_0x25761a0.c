// Function: sub_25761A0
// Address: 0x25761a0
//
char __fastcall sub_25761A0(__int64 a1, const void **a2)
{
  __int64 (*v2)(void); // rax
  char result; // al
  unsigned int v4; // eax
  __int64 (__fastcall *v5)(__int64); // rax

  v2 = *(__int64 (**)(void))(*(_QWORD *)a1 + 16LL);
  if ( (char *)v2 == (char *)sub_2505E40 )
    result = *(_BYTE *)(a1 + 17);
  else
    result = v2();
  if ( result )
  {
    sub_2575FB0((_DWORD *)(a1 + 24), a2);
    v4 = *(_DWORD *)(a1 + 64);
    if ( v4 < unk_4FEF868 )
    {
      result = v4 == 0;
      *(_BYTE *)(a1 + 200) &= result;
    }
    else
    {
      v5 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 40LL);
      if ( v5 == sub_2534FB0 )
      {
        result = *(_BYTE *)(a1 + 16);
        *(_BYTE *)(a1 + 17) = result;
      }
      else
      {
        return v5(a1);
      }
    }
  }
  return result;
}
