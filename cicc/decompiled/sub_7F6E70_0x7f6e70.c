// Function: sub_7F6E70
// Address: 0x7f6e70
//
_BYTE *__fastcall sub_7F6E70(__int64 a1, int *a2)
{
  _BYTE *result; // rax
  _QWORD *v3; // r14
  _BYTE *v4; // rax
  _BYTE *v5; // rax
  _BYTE *v6; // rax
  __int64 v7; // r8
  __int64 v8; // r9

  result = &dword_4F06888;
  if ( !dword_4F06888 )
  {
    if ( unk_4F06884 )
    {
      v3 = sub_73A830(1, 5u);
      v6 = sub_731250(a1);
    }
    else
    {
      v3 = sub_73A830(1, 0);
      v4 = sub_73E230(a1, 0);
      v5 = sub_7E23D0(v4);
      v6 = sub_73DCD0(v5);
    }
    return sub_7E6A80(v6, 0x49u, (__int64)v3, a2, v7, v8);
  }
  return result;
}
