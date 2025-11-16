// Function: sub_5F77E0
// Address: 0x5f77e0
//
__int64 __fastcall sub_5F77E0(__int64 *a1, _QWORD *a2)
{
  int v3; // eax
  unsigned __int64 v4; // rsi
  unsigned __int64 i; // rdx
  unsigned int v6; // edx
  __int64 v7; // rax
  __int64 v8; // r14
  __int64 result; // rax
  char v10; // r8
  __int64 v11; // rdi
  __int64 v12; // [rsp+8h] [rbp-38h]
  _DWORD v13[9]; // [rsp+1Ch] [rbp-24h] BYREF

  v3 = sub_85E8D0();
  if ( v3 == -1 )
    goto LABEL_13;
  v4 = *(_QWORD *)(qword_4F04C68[0] + 776LL * v3 + 216);
  for ( i = v4 >> 3; ; LODWORD(i) = v6 + 1 )
  {
    v6 = qword_4F04C10[1] & i;
    v7 = *qword_4F04C10 + 16LL * v6;
    if ( v4 == *(_QWORD *)v7 )
      break;
    if ( !*(_QWORD *)v7 )
      goto LABEL_13;
  }
  v8 = *(_QWORD *)(v7 + 8);
  if ( !v8 )
  {
LABEL_13:
    sub_6854C0(2644, a2, *a1);
    return 0;
  }
  result = (__int64)sub_5EA830(*(__int64 ****)(v7 + 8), 0, a1);
  if ( !result )
  {
    v10 = *(_BYTE *)(v8 + 24);
    if ( (v10 & 0x10) != 0 )
    {
      result = sub_5F72B0(v8, 0, a1, 1, (v10 & 0x20) != 0, a2, v13);
      v11 = 1877;
      if ( !v13[0] )
        return result;
    }
    else
    {
      v11 = 1735;
    }
    v12 = result;
    sub_6851C0(v11, a2);
    return v12;
  }
  return result;
}
