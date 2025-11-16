// Function: sub_18EC750
// Address: 0x18ec750
//
char __fastcall sub_18EC750(__int64 a1, __int64 a2, __int64 a3, unsigned int a4)
{
  __int64 v5; // rdi
  __int64 v6; // r8
  __int64 v7; // rax
  __int64 *v8; // r8
  unsigned int v10; // [rsp+4h] [rbp-2Ch]
  __int64 v11; // [rsp+8h] [rbp-28h]
  __int64 v12; // [rsp+18h] [rbp-18h]

  if ( (*(_BYTE *)(a3 + 23) & 0x40) != 0 )
    v5 = *(_QWORD *)(a3 - 8);
  else
    v5 = a3 - 24LL * (*(_DWORD *)(a3 + 20) & 0xFFFFFFF);
  v6 = *(_QWORD *)(v5 + 24LL * a4);
  LODWORD(v7) = *(unsigned __int8 *)(v6 + 16);
  if ( (_BYTE)v7 == 13 )
    goto LABEL_4;
  if ( (unsigned __int8)v7 <= 0x17u )
  {
    v10 = a4;
    v11 = a3;
    if ( (_BYTE)v7 == 5 )
    {
      v12 = *(_QWORD *)(v5 + 24LL * a4);
      LOBYTE(v7) = sub_1594510(v6);
      if ( (_BYTE)v7 )
      {
        LODWORD(v7) = -3 * (*(_DWORD *)(v12 + 20) & 0xFFFFFFF);
        v6 = *(_QWORD *)(v12 - 24LL * (*(_DWORD *)(v12 + 20) & 0xFFFFFFF));
        if ( *(_BYTE *)(v6 + 16) == 13 )
        {
          a4 = v10;
          a3 = v11;
LABEL_4:
          LOBYTE(v7) = (unsigned __int8)sub_18EC2C0(a1, a2, a3, a4, v6);
        }
      }
    }
  }
  else
  {
    LODWORD(v7) = v7 - 60;
    if ( (unsigned int)v7 <= 0xC )
    {
      if ( (*(_BYTE *)(v6 + 23) & 0x40) != 0 )
      {
        v8 = *(__int64 **)(v6 - 8);
      }
      else
      {
        v7 = 24LL * (*(_DWORD *)(v6 + 20) & 0xFFFFFFF);
        v8 = (__int64 *)(v6 - v7);
      }
      v6 = *v8;
      if ( *(_BYTE *)(v6 + 16) == 13 )
        goto LABEL_4;
    }
  }
  return v7;
}
