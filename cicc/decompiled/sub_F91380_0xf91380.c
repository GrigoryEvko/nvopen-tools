// Function: sub_F91380
// Address: 0xf91380
//
__int64 __fastcall sub_F91380(char *a1)
{
  char v1; // al
  char *v2; // r12
  __int64 result; // rax
  _BYTE v4[16]; // [rsp+0h] [rbp-30h] BYREF
  __int64 (__fastcall *v5)(_BYTE *, _BYTE *, __int64); // [rsp+10h] [rbp-20h]

  v1 = *a1;
  if ( *a1 == 32 )
  {
LABEL_2:
    v2 = (char *)**((_QWORD **)a1 - 1);
    if ( !v2 )
      BUG();
    if ( (unsigned __int8)*v2 > 0x1Cu )
      goto LABEL_4;
    return sub_B43D60(a1);
  }
  if ( v1 != 31 )
  {
    if ( v1 != 33 )
      return sub_B43D60(a1);
    goto LABEL_2;
  }
  if ( (*((_DWORD *)a1 + 1) & 0x7FFFFFF) != 3 )
    return sub_B43D60(a1);
  v2 = (char *)*((_QWORD *)a1 - 12);
  if ( (unsigned __int8)*v2 <= 0x1Cu )
    return sub_B43D60(a1);
LABEL_4:
  sub_B43D60(a1);
  v5 = 0;
  sub_F5CAB0(v2, 0, 0, (__int64)v4);
  result = (__int64)v5;
  if ( v5 )
    return v5(v4, v4, 3);
  return result;
}
