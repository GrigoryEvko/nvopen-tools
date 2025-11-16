// Function: sub_AC51A0
// Address: 0xac51a0
//
__int64 __fastcall sub_AC51A0(__int64 a1, __int64 a2)
{
  bool v2; // zf
  __int64 result; // rax
  unsigned int v4; // edx
  unsigned int v5; // edx
  __int64 v6; // rdx
  unsigned int v7; // edx
  __int64 v8; // [rsp-20h] [rbp-20h]

  v2 = *(_BYTE *)(a2 + 72) == 0;
  *(_BYTE *)(a1 + 32) = 0;
  result = a1;
  if ( !v2 )
  {
    v4 = *(_DWORD *)(a2 + 48);
    *(_DWORD *)(a1 + 8) = v4;
    if ( v4 > 0x40 )
    {
      sub_C43780(a1, a2 + 40);
      result = a1;
      v7 = *(_DWORD *)(a2 + 64);
      *(_DWORD *)(a1 + 24) = v7;
      if ( v7 <= 0x40 )
        goto LABEL_4;
    }
    else
    {
      *(_QWORD *)a1 = *(_QWORD *)(a2 + 40);
      v5 = *(_DWORD *)(a2 + 64);
      *(_DWORD *)(a1 + 24) = v5;
      if ( v5 <= 0x40 )
      {
LABEL_4:
        v6 = *(_QWORD *)(a2 + 56);
        *(_BYTE *)(result + 32) = 1;
        *(_QWORD *)(result + 16) = v6;
        return result;
      }
    }
    v8 = result;
    sub_C43780(result + 16, a2 + 56);
    *(_BYTE *)(v8 + 32) = 1;
    return v8;
  }
  return result;
}
