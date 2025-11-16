// Function: sub_1340EA0
// Address: 0x1340ea0
//
__int64 __fastcall sub_1340EA0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6, __int64 a7)
{
  __int64 v11; // r11
  __int64 result; // rax
  __int64 v13; // [rsp+10h] [rbp-40h]
  int v14; // [rsp+1Ch] [rbp-34h]

  v11 = qword_50579C0[(unsigned int)a7];
  if ( !v11 )
    return sub_13468F0(a2, a3, a4, a5, a6);
  if ( *(_DWORD *)(v11 + 10520) == 1 )
  {
    result = sub_1346420(a1, qword_50579C0[(unsigned int)a7], a2, a3, a4, a5, a6);
    if ( result )
      return result;
    return sub_13468F0(a2, a3, a4, a5, a6);
  }
  v13 = qword_50579C0[(unsigned int)a7];
  v14 = *(_DWORD *)(v11 + 10520);
  result = sub_13468F0(a2, a3, a4, a5, a6);
  if ( !result && v14 == 2 )
    return sub_1346420(a1, v13, a2, a3, a4, a5, a6);
  return result;
}
