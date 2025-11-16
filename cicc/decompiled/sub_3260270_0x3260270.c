// Function: sub_3260270
// Address: 0x3260270
//
__int64 __fastcall sub_3260270(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, char a6, __int64 a7)
{
  __int64 v10; // rax
  __int64 v11; // r8
  __int16 v12; // dx
  __int64 v13; // rax
  __int64 result; // rax
  char v15; // al
  __int64 (*v16)(); // rax
  char v17; // al
  char v18; // [rsp+Ch] [rbp-44h]
  char v19; // [rsp+Ch] [rbp-44h]
  unsigned int v20; // [rsp+10h] [rbp-40h] BYREF
  __int64 v21; // [rsp+18h] [rbp-38h]

  v10 = *(_QWORD *)(a2 + 48) + 16LL * (unsigned int)a3;
  v11 = a7;
  v12 = *(_WORD *)v10;
  v13 = *(_QWORD *)(v10 + 8);
  LOWORD(v20) = v12;
  v21 = v13;
  if ( v12 )
  {
    if ( (unsigned __int16)(v12 - 10) > 6u
      && (unsigned __int16)(v12 - 126) > 0x31u
      && (unsigned __int16)(v12 - 208) > 0x14u )
    {
      return 0;
    }
  }
  else
  {
    v18 = a6;
    v15 = sub_3007030((__int64)&v20);
    a6 = v18;
    v11 = a7;
    if ( !v15 )
      return 0;
  }
  if ( a6 >= 0 && (*(_BYTE *)(*(_QWORD *)a1 + 864LL) & 0x10) == 0 )
    return 0;
  v16 = *(__int64 (**)())(*(_QWORD *)v11 + 1232LL);
  if ( v16 != sub_2FE3370 )
  {
    v19 = a6;
    v17 = ((__int64 (__fastcall *)(__int64, _QWORD, __int64))v16)(v11, v20, v21);
    a6 = v19;
    if ( !v17 )
      return 0;
  }
  result = 1;
  if ( (a6 & 0x20) == 0 )
  {
    if ( (unsigned __int8)sub_33CE830(a1, a4, a5, 0, 0) )
      return sub_33CE830(a1, a2, a3, 0, 0);
    return 0;
  }
  return result;
}
