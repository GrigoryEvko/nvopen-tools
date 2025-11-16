// Function: sub_26CB0E0
// Address: 0x26cb0e0
//
__int64 __fastcall sub_26CB0E0(__int64 a1, _QWORD *a2, unsigned __int8 *a3)
{
  int v3; // eax
  unsigned __int64 v4; // rax
  __int64 v5; // rcx
  __int64 (__fastcall **v6)(); // rax
  __int64 v8; // rax
  bool v9; // al
  _QWORD *v10; // rax
  __int64 v11; // [rsp+8h] [rbp-18h]

  if ( !unk_4F838D4 )
  {
    if ( *((_QWORD *)a3 + 6) )
    {
      v3 = *a3;
      if ( (_BYTE)v3 != 31 )
      {
        if ( (_BYTE)v3 == 85 )
        {
          v8 = *((_QWORD *)a3 - 4);
          if ( !v8 || *(_BYTE *)v8 || *(_QWORD *)(v8 + 24) != *((_QWORD *)a3 + 10) || (*(_BYTE *)(v8 + 33) & 0x20) == 0 )
          {
            if ( unk_4F838D3 )
            {
LABEL_9:
              sub_26C6990(a1, a2, (__int64)a3);
              return a1;
            }
LABEL_18:
            v11 = (__int64)a3;
            v9 = sub_B491E0((__int64)a3);
            a3 = (unsigned __int8 *)v11;
            if ( !v9 )
            {
              v10 = sub_26CAE90(a2, v11);
              a3 = (unsigned __int8 *)v11;
              if ( v10 )
              {
                *(_BYTE *)(a1 + 16) &= ~1u;
                *(_QWORD *)a1 = 0;
                return a1;
              }
            }
            goto LABEL_9;
          }
        }
        else if ( (_BYTE)v3 != 84 )
        {
          if ( unk_4F838D3 )
            goto LABEL_9;
          v4 = (unsigned int)(v3 - 34);
          if ( (unsigned __int8)v4 > 0x33u )
            goto LABEL_9;
          v5 = 0x8000000000041LL;
          if ( !_bittest64(&v5, v4) )
            goto LABEL_9;
          goto LABEL_18;
        }
      }
    }
    v6 = sub_2241E40();
    *(_BYTE *)(a1 + 16) |= 1u;
    *(_DWORD *)a1 = 0;
    *(_QWORD *)(a1 + 8) = v6;
    return a1;
  }
  sub_26C63A0(a1, a2, (__int64)a3);
  return a1;
}
