// Function: sub_7A9360
// Address: 0x7a9360
//
__int64 __fastcall sub_7A9360(__int64 **a1, __int64 *a2)
{
  __int64 v2; // rax
  unsigned int v3; // edx
  unsigned __int64 v4; // r14
  unsigned int v6[9]; // [rsp+Ch] [rbp-24h] BYREF

  if ( unk_4F06AB4 && (unsigned int)sub_7A80B0(a2[5]) )
  {
    sub_7A87A0((__int64)a1, (__int64)a2);
    if ( unk_4D0424C )
LABEL_8:
      sub_7A8F40(0, (__int64)a2, a1);
  }
  else
  {
    v2 = *(_QWORD *)(a2[5] + 168);
    v3 = *(_DWORD *)(v2 + 40);
    v4 = *(_QWORD *)(v2 + 32);
    v6[0] = v3;
    if ( unk_4F06A74 )
    {
      sub_7A65D0(v6, a2[7]);
      v3 = v6[0];
    }
    a2[13] = sub_7A8C40((__int64)a1, v4, v3, (__int64)a2);
    if ( unk_4D0424C )
      goto LABEL_8;
  }
  return sub_7A6830((__int64)a1, (__int64)a2);
}
