// Function: sub_5CACA0
// Address: 0x5caca0
//
__int64 __fastcall sub_5CACA0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 *a5)
{
  __int64 v5; // r13
  char v6; // al
  __int64 v8; // rax
  char *v9; // rax
  _DWORD v13[9]; // [rsp+2Ch] [rbp-24h] BYREF

  v5 = *(_QWORD *)(a1 + 40);
  v6 = *(_BYTE *)(v5 + 173);
  if ( v6 == 12 || !v6 )
    return 0;
  v13[0] = 0;
  if ( v6 == 1 && (unsigned int)sub_8D2930(*(_QWORD *)(v5 + 128)) )
  {
    v8 = sub_620FA0(v5, v13);
    *a5 = v8;
    if ( v8 > a4 || v8 < a3 || v13[0] )
    {
      v9 = sub_5C79F0(a2);
      sub_6851A0(1099, a1 + 24, v9);
      *(_BYTE *)(a2 + 8) = 0;
      return 0;
    }
    else
    {
      return 1;
    }
  }
  else
  {
    sub_6851C0(661, a1 + 24);
    *(_BYTE *)(a2 + 8) = 0;
    return 0;
  }
}
