// Function: sub_2E7BE50
// Address: 0x2e7be50
//
unsigned __int64 __fastcall sub_2E7BE50(_QWORD *a1, int a2)
{
  unsigned __int64 result; // rax
  __int64 v3; // rcx
  unsigned __int64 v4; // rdx

  result = a1[8];
  if ( !result )
  {
    v3 = a1[16];
    a1[26] += 32LL;
    v4 = (v3 + 15) & 0xFFFFFFFFFFFFFFF0LL;
    if ( a1[17] >= v4 + 32 && v3 )
    {
      a1[16] = v4 + 32;
      if ( !v4 )
      {
LABEL_6:
        a1[8] = result;
        return result;
      }
    }
    else
    {
      v4 = sub_9D1E70((__int64)(a1 + 16), 32, 32, 4);
    }
    *(_DWORD *)v4 = a2;
    result = v4;
    *(_QWORD *)(v4 + 8) = 0;
    *(_QWORD *)(v4 + 16) = 0;
    *(_QWORD *)(v4 + 24) = 0;
    goto LABEL_6;
  }
  return result;
}
