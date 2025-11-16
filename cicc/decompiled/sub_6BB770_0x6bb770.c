// Function: sub_6BB770
// Address: 0x6bb770
//
__int64 __fastcall sub_6BB770(__int64 a1, unsigned int a2, int a3, _DWORD *a4)
{
  char v5; // r13
  __int64 v6; // r12
  __int64 v8; // rdi
  _DWORD *v9; // r15
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9
  _QWORD v15[7]; // [rsp+18h] [rbp-38h] BYREF

  if ( a4 )
    *a4 = 0;
  v5 = a3 != 0;
  if ( *(_QWORD *)(a1 + 328) )
  {
    v6 = sub_6E1C80(a1 + 328);
    v5 &= v6 != 0;
  }
  else if ( (*(_BYTE *)(a1 + 129) & 2) != 0 )
  {
    if ( dword_4F04C64 != -1 && (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 7) & 1) != 0 )
    {
      v15[0] = *(_QWORD *)&dword_4F063F8;
      sub_6BB600(a1 + 328, a2);
      v6 = sub_6E1C80(a1 + 328);
      if ( v6 )
      {
        if ( (*(_BYTE *)(a1 + 129) & 4) != 0 )
        {
          v8 = *(_QWORD *)(a1 + 328);
          if ( v8 )
          {
            v9 = (_DWORD *)sub_6E1A20(v8);
            if ( (unsigned int)sub_6E5430(v8, a2, v10, v11, v12, v13) )
              sub_6851C0(0x832u, v9);
            sub_6E1BF0(a1 + 328);
          }
        }
      }
      else
      {
        if ( a4 )
        {
          *a4 = 1;
          return v6;
        }
        sub_6851C0(0x7CCu, v15);
        v6 = sub_6E2F40(0);
        sub_6E6260(*(_QWORD *)(v6 + 24) + 8LL);
        *(_QWORD *)(*(_QWORD *)(v6 + 24) + 76LL) = v15[0];
      }
    }
    else
    {
      v6 = sub_6BB5A0(a2, 0);
      v5 &= v6 != 0;
    }
  }
  else
  {
    v6 = sub_6A2C00(a2, 1u);
    v5 &= v6 != 0;
  }
  if ( v5 && *(_BYTE *)(v6 + 8) == 1 && (*(_BYTE *)(a1 + 179) & 0x20) == 0 )
    *(_BYTE *)(v6 + 9) |= 0x10u;
  return v6;
}
