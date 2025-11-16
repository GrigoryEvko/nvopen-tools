// Function: sub_2D010F0
// Address: 0x2d010f0
//
__int64 __fastcall sub_2D010F0(__int64 a1, unsigned __int64 a2)
{
  int v2; // ebx
  unsigned int v3; // r14d
  unsigned int v5; // eax
  int v6; // edx
  unsigned int v7; // [rsp+4h] [rbp-2Ch] BYREF
  __int64 v8[5]; // [rsp+8h] [rbp-28h] BYREF

  v2 = sub_2D00C30((unsigned int *)a1, (_BYTE *)a2);
  if ( *(_BYTE *)(*(_QWORD *)(*(_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)) + 8LL) + 8LL) != 14 )
    return 0;
  v7 = 0;
  v3 = sub_2D00530(a1, a2, v8, &v7);
  if ( (_BYTE)v3 )
  {
    v5 = sub_2D00C30((unsigned int *)a1, (_BYTE *)v8[0]);
    v6 = *(_DWORD *)a1;
    if ( v5 == *(_DWORD *)a1 )
    {
      if ( v6 != v2 )
      {
LABEL_7:
        sub_2D00AD0((_QWORD *)a1, a2, v6);
        return v3;
      }
    }
    else
    {
      v6 = sub_2D00850(a1, v7, v5);
      if ( v6 != v2 )
        goto LABEL_7;
    }
    return 0;
  }
  sub_2D00AD0((_QWORD *)a1, a2, *(_DWORD *)(a1 + 4));
  return v3;
}
