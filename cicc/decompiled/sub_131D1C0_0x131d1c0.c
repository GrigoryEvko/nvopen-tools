// Function: sub_131D1C0
// Address: 0x131d1c0
//
bool __fastcall sub_131D1C0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, unsigned __int64 a5, unsigned __int64 a6)
{
  unsigned __int64 v7; // r12
  __int64 (__fastcall *v9)(__int64, const char *); // rax
  unsigned __int64 v10; // rax

  if ( a3 )
  {
    *(_QWORD *)a2 = a3;
    *(_QWORD *)(a2 + 8) = a4;
    if ( a5 )
    {
LABEL_3:
      *(_QWORD *)(a2 + 16) = a5;
      *(_BYTE *)(a2 + 40) = 0;
      goto LABEL_4;
    }
  }
  else
  {
    v9 = (__int64 (__fastcall *)(__int64, const char *))unk_505F9C0;
    if ( !unk_505F9C0 )
      v9 = sub_130AA00;
    *(_QWORD *)a2 = v9;
    *(_QWORD *)(a2 + 8) = a4;
    if ( a5 )
      goto LABEL_3;
  }
  v10 = sub_131C910(a1, a6);
  *(_BYTE *)(a2 + 40) = 1;
  *(_QWORD *)(a2 + 16) = v10;
  a5 = v10;
  if ( !v10 )
  {
    v7 = 0;
    goto LABEL_5;
  }
LABEL_4:
  v7 = a6 - 1;
LABEL_5:
  *(_QWORD *)(a2 + 24) = v7;
  *(_QWORD *)(a2 + 32) = 0;
  return a5 == 0;
}
