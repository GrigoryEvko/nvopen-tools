// Function: sub_27366A0
// Address: 0x27366a0
//
char __fastcall sub_27366A0(__int64 a1, __int64 a2, __int64 a3, unsigned int a4)
{
  __int64 v5; // rdi
  unsigned __int8 *v6; // r8
  __int64 v7; // rax
  unsigned __int8 *v8; // r8
  unsigned int v10; // [rsp+0h] [rbp-30h]
  unsigned __int8 *v11; // [rsp+0h] [rbp-30h]
  __int64 v12; // [rsp+8h] [rbp-28h]
  unsigned int v13; // [rsp+8h] [rbp-28h]
  __int64 v14; // [rsp+10h] [rbp-20h]
  unsigned __int8 *v15; // [rsp+18h] [rbp-18h]

  if ( (*(_BYTE *)(a3 + 7) & 0x40) != 0 )
    v5 = *(_QWORD *)(a3 - 8);
  else
    v5 = a3 - 32LL * (*(_DWORD *)(a3 + 4) & 0x7FFFFFF);
  v6 = *(unsigned __int8 **)(v5 + 32LL * a4);
  LODWORD(v7) = *v6;
  if ( (_BYTE)v7 == 17 )
    goto LABEL_4;
  if ( (unsigned __int8)v7 <= 0x1Cu )
  {
    if ( (_BYTE)v7 == 5 )
    {
      if ( (_BYTE)qword_4FF9F28 && *((_WORD *)v6 + 1) == 34 )
      {
        v11 = *(unsigned __int8 **)(v5 + 32LL * a4);
        v13 = a4;
        v14 = a3;
        sub_27362B0(a1, a2, a3, a4, (__int64)v6);
        v6 = v11;
        a4 = v13;
        a3 = v14;
      }
      v10 = a4;
      v12 = a3;
      v15 = v6;
      LOBYTE(v7) = sub_AC35E0((__int64)v6);
      if ( (_BYTE)v7 )
      {
        LODWORD(v7) = -32 * (*((_DWORD *)v15 + 1) & 0x7FFFFFF);
        v6 = *(unsigned __int8 **)&v15[-32 * (*((_DWORD *)v15 + 1) & 0x7FFFFFF)];
        if ( *v6 == 17 )
        {
          a4 = v10;
          a3 = v12;
LABEL_4:
          LOBYTE(v7) = sub_27334C0(a1, a2, a3, a4, (__int64)v6);
        }
      }
    }
  }
  else
  {
    LODWORD(v7) = v7 - 67;
    if ( (unsigned int)v7 <= 0xC )
    {
      if ( (v6[7] & 0x40) != 0 )
      {
        v8 = (unsigned __int8 *)*((_QWORD *)v6 - 1);
      }
      else
      {
        v7 = 32LL * (*((_DWORD *)v6 + 1) & 0x7FFFFFF);
        v8 = &v6[-v7];
      }
      v6 = *(unsigned __int8 **)v8;
      if ( *v6 == 17 )
        goto LABEL_4;
    }
  }
  return v7;
}
