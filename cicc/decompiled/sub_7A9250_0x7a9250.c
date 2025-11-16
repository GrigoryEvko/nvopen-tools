// Function: sub_7A9250
// Address: 0x7a9250
//
__int64 __fastcall sub_7A9250(__int64 a1, __int64 *a2)
{
  __int64 v2; // rax
  unsigned int v3; // edx
  unsigned __int64 v4; // r14
  unsigned __int64 v5; // rax
  unsigned __int64 v6; // rax
  unsigned int v8[9]; // [rsp+Ch] [rbp-24h] BYREF

  if ( unk_4F06AB4 && (unsigned int)sub_7A80B0(a2[5]) )
  {
    sub_7A87A0(a1, (__int64)a2);
    v5 = a2[13];
  }
  else
  {
    v2 = *(_QWORD *)(a2[5] + 168);
    v3 = *(_DWORD *)(v2 + 40);
    v4 = *(_QWORD *)(v2 + 32);
    v8[0] = v3;
    if ( unk_4F06A74 )
    {
      sub_7A65D0(v8, a2[7]);
      v3 = v8[0];
    }
    v5 = sub_7A8C40(a1, v4, v3, (__int64)a2);
    a2[13] = v5;
  }
  v6 = *(_QWORD *)(*(_QWORD *)(a2[5] + 168) + 32LL) + v5;
  if ( *(_QWORD *)(a1 + 56) + 1LL < v6 )
    *(_QWORD *)(a1 + 56) = v6 - 1;
  if ( unk_4D0424C )
    sub_7A8F40(0, (__int64)a2, (__int64 **)a1);
  return sub_7A6830(a1, (__int64)a2);
}
