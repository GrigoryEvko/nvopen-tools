// Function: sub_72AEB0
// Address: 0x72aeb0
//
__int64 __fastcall sub_72AEB0(__int64 a1, _DWORD *a2)
{
  __int64 result; // rax
  __int64 v3; // r12
  int i; // eax
  int v5; // edx
  __int64 j; // rax
  __int64 v7; // rcx
  __int64 v8; // rax
  __int64 v9; // rax

  result = (__int64)&dword_4F077C4;
  if ( dword_4F077C4 == 2 && (*(_BYTE *)(a1 + 32) & 1) == 0 )
  {
    v3 = *(_QWORD *)(a1 + 8);
    for ( i = *(unsigned __int8 *)(v3 + 140); (_BYTE)i == 12; i = *(unsigned __int8 *)(v3 + 140) )
      v3 = *(_QWORD *)(v3 + 160);
    result = (unsigned int)(i - 9);
    if ( (unsigned __int8)result <= 2u )
    {
      v5 = sub_8D23B0(v3);
      if ( v5 )
        return sub_880320(v3, 0, a1, 3, a2);
      for ( j = v3; *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
        ;
      result = *(_QWORD *)(*(_QWORD *)j + 96LL);
      if ( (*(_BYTE *)(result + 177) & 0x40) == 0
        || !qword_4D0495C
        && (result = *(_QWORD *)(result + 24)) != 0
        && ((result = *(_QWORD *)(result + 88), (*(_BYTE *)(result + 194) & 8) == 0)
         || (*(_BYTE *)(result + 206) & 0x10) != 0) )
      {
        *(_BYTE *)(a1 + 32) |= 1u;
        if ( (*(_BYTE *)(v3 + 176) & 0x20) != 0 )
        {
          result = 8;
          if ( !dword_4F077BC )
            goto LABEL_24;
          if ( unk_4F04C48 == -1
            || (v7 = qword_4F04C68[0], (*(_BYTE *)(qword_4F04C68[0] + 776LL * unk_4F04C48 + 10) & 1) == 0) )
          {
            result = 8;
            if ( dword_4F04C44 == -1 )
              goto LABEL_24;
            v7 = qword_4F04C68[0];
          }
          v8 = 776LL * dword_4F04C64;
          if ( *(_BYTE *)(v7 + v8 + 4) == 1 )
          {
            v9 = v7 + v8 - 772;
            do
            {
              v9 -= 776;
              ++v5;
            }
            while ( *(_BYTE *)(v9 + 776) == 1 );
            LOBYTE(v9) = v5 == 1;
            result = (unsigned int)(3 * v9 + 5);
          }
          else
          {
            result = 8;
          }
LABEL_24:
          if ( *a2 )
          {
            if ( !dword_4D041AC )
              return sub_5EB950(result, 603, v3, (__int64)a2);
          }
        }
      }
    }
  }
  return result;
}
