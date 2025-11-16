// Function: sub_5D2430
// Address: 0x5d2430
//
__int64 __fastcall sub_5D2430(__int64 a1, __int64 a2)
{
  __int64 v2; // rbx
  _QWORD *v3; // rax
  __int64 *v4; // rax
  __int64 v5; // rbx
  __int64 v6; // r14
  unsigned __int64 v7; // rax
  _DWORD v9[9]; // [rsp+Ch] [rbp-24h] BYREF

  if ( (*(_BYTE *)(a2 + 89) & 4) != 0 )
    sub_6849F0(7, 3535, a1 + 56, "__launch_bounds__");
  v2 = *(_QWORD *)(a1 + 32);
  if ( v2 )
  {
    v3 = *(_QWORD **)(a2 + 328);
    if ( !v3 )
    {
      v3 = (_QWORD *)sub_725F60();
      *(_QWORD *)(a2 + 328) = v3;
    }
    *v3 = *(_QWORD *)(v2 + 40);
    v4 = *(__int64 **)v2;
    if ( *(_QWORD *)v2 )
    {
      *(_QWORD *)(*(_QWORD *)(a2 + 328) + 8LL) = v4[5];
      v5 = *v4;
      if ( *v4 )
      {
        if ( unk_4D045E8 <= 0x59u )
          sub_684AA0(7, 3704, a1 + 56);
        if ( !(unsigned int)sub_5D19D0(a1) )
        {
          v6 = *(_QWORD *)(v5 + 40);
          if ( v6 )
          {
            if ( (int)sub_6210B0(*(_QWORD *)(v5 + 40), 0) < 0 )
            {
              sub_684AA0(7, 3705, a1 + 56);
            }
            else
            {
              v7 = sub_620FD0(v6, v9);
              if ( v9[0] || v7 > 0x7FFFFFFF )
              {
                sub_684AA0(7, 3706, a1 + 56);
              }
              else if ( v7 )
              {
                *(_DWORD *)(*(_QWORD *)(a2 + 328) + 16LL) = v7;
              }
            }
          }
        }
      }
    }
  }
  return a2;
}
