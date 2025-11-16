// Function: sub_6EC9C0
// Address: 0x6ec9c0
//
__int64 __fastcall sub_6EC9C0(__int64 *a1)
{
  __int64 v1; // r12
  __int64 result; // rax
  int v3; // ecx
  __int64 *v4; // rax
  __int64 v5; // [rsp-10h] [rbp-40h]
  __int64 v6[5]; // [rsp+8h] [rbp-28h] BYREF

  v1 = *a1;
  for ( result = a1[18]; *(_BYTE *)(v1 + 140) == 12; v1 = *(_QWORD *)(v1 + 160) )
    ;
  if ( *(_BYTE *)(result + 24) == 1 && *(_BYTE *)(result + 56) == 91 )
  {
    result = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(*(_QWORD *)(result + 72) + 16LL) + 24LL) - 5;
    if ( (unsigned __int8)result <= 1u )
    {
      if ( HIDWORD(qword_4F077B4) | unk_4D04460 && (unsigned int)sub_8D3A70(v1) )
      {
        v3 = 128;
        if ( HIDWORD(qword_4F077B4) )
        {
          if ( !(_DWORD)qword_4F077B4 )
            v3 = qword_4F077A8 == 0 ? 128 : 131200;
        }
        sub_8470D0((_DWORD)a1, v1, 1, v3, 0, 0, (__int64)v6);
        v4 = (__int64 *)sub_6EC670(*a1, v6[0], 0, 0);
        sub_6E7170(v4, (__int64)a1);
        return v5;
      }
      else
      {
        return sub_8443E0(a1, 0, 0);
      }
    }
  }
  return result;
}
