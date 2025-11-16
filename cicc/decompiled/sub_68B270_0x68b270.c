// Function: sub_68B270
// Address: 0x68b270
//
void __fastcall sub_68B270(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v6; // rdi
  _QWORD *v8; // rsi
  _DWORD v9[5]; // [rsp+Ch] [rbp-14h] BYREF

  v6 = *a1;
  if ( v6 != a2 && !(unsigned int)sub_8D97D0(v6, a2, 0, a4, a5) && (unsigned int)sub_8D2930(a2) )
  {
    sub_6E6B60(a1, 0);
    v8 = 0;
    if ( *((_BYTE *)a1 + 16) == 2 )
      v8 = a1 + 18;
    if ( (unsigned int)sub_8D67E0(*a1, v8, a2, 0, v9) )
    {
      sub_6E5ED0(v9[0], (char *)a1 + 68, *a1, a2);
      sub_6E6260(a1);
    }
  }
}
