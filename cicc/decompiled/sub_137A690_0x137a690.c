// Function: sub_137A690
// Address: 0x137a690
//
__int64 __fastcall sub_137A690(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  unsigned int v4; // r8d
  __int64 v6; // rax
  int v7; // edi
  unsigned int v8; // eax
  __int64 v9; // rdi
  int v10; // r14d
  int v11; // r15d
  int v12[9]; // [rsp+Ch] [rbp-24h] BYREF

  v3 = sub_157EBA0(a2);
  v4 = 0;
  if ( *(_BYTE *)(v3 + 16) == 26 && (*(_DWORD *)(v3 + 20) & 0xFFFFFFF) == 3 )
  {
    v6 = *(_QWORD *)(v3 - 72);
    if ( *(_BYTE *)(v6 + 16) == 76 )
    {
      v7 = *(unsigned __int16 *)(v6 + 18);
      v8 = v7 & 0xFFFF7FF7;
      v9 = v7 & 0xFFFF7FFF;
      LOBYTE(v4) = v8 == 1 || v8 == 6;
      if ( (_BYTE)v4 )
      {
        if ( !(unsigned __int8)sub_15FF820(v9) )
        {
LABEL_7:
          v10 = 1;
          v11 = 0;
LABEL_8:
          sub_16AF710(v12, 20, 32);
          sub_1379150(a1, a2, v11, v12[0]);
          sub_1379150(a1, a2, v10, 0x80000000 - v12[0]);
          return 1;
        }
      }
      else
      {
        if ( (_DWORD)v9 == 7 )
          goto LABEL_7;
        if ( (_DWORD)v9 != 8 )
          return v4;
      }
      v10 = 0;
      v11 = 1;
      goto LABEL_8;
    }
  }
  return v4;
}
