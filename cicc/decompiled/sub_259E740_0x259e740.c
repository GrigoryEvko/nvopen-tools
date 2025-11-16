// Function: sub_259E740
// Address: 0x259e740
//
__int64 __fastcall sub_259E740(__int64 a1, __int64 a2)
{
  char *v2; // r13
  char v3; // al
  unsigned __int64 v5; // rax
  __int64 v6; // rax
  __int64 v7; // r15
  unsigned __int64 v8; // rax
  __m128i v9[3]; // [rsp+0h] [rbp-30h] BYREF

  v2 = (char *)sub_250D070((_QWORD *)(a1 + 72));
  v3 = *v2;
  if ( (unsigned __int8)*v2 <= 0x1Cu )
  {
    v2 = 0;
    goto LABEL_10;
  }
  if ( v3 == 62 )
  {
    if ( sub_2574220(a1, a2, (__int64)v2, 0) )
      return 1;
    goto LABEL_8;
  }
  if ( v3 != 64 )
  {
LABEL_10:
    if ( sub_259E650(a1, a2, (unsigned __int64)v2) )
    {
      v8 = sub_250D070((_QWORD *)(a1 + 72));
      if ( (unsigned __int8)sub_254E2A0(a1, a2, v8) )
        return 1;
    }
    goto LABEL_8;
  }
  v5 = sub_B43CB0((__int64)v2);
  sub_250D230((unsigned __int64 *)v9, v5, 4, 0);
  v6 = sub_2567630(a2, v9, a1, 2, 0);
  v7 = v6;
  if ( v6 && (*(unsigned __int8 (__fastcall **)(__int64, char *))(*(_QWORD *)v6 + 152LL))(v6, v2) )
  {
    sub_250ED80(a2, v7, a1, 1);
    return 1;
  }
LABEL_8:
  *(_BYTE *)(a1 + 97) = *(_BYTE *)(a1 + 96);
  return 0;
}
