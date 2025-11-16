// Function: sub_7BEFD0
// Address: 0x7befd0
//
__int64 __fastcall sub_7BEFD0(__int64 a1, __int64 a2)
{
  char v2; // dl
  char v3; // dl
  char v4; // al
  _BOOL4 v5; // eax
  char v6; // dl
  __int64 result; // rax
  __int64 v8; // rdx
  __int64 *v9[3]; // [rsp+8h] [rbp-18h] BYREF

  sub_727530(a2);
  *(_QWORD *)(a2 + 8) = *(_QWORD *)(a1 + 48);
  v2 = ((*(_BYTE *)(a1 + 16) & 2) != 0) | *(_BYTE *)(a2 + 33) & 0xFE;
  *(_BYTE *)(a2 + 33) = v2;
  v3 = (2 * (*(_BYTE *)(a1 + 18) & 1)) | v2 & 0xFD;
  *(_BYTE *)(a2 + 33) = v3;
  v4 = v3 & 0xFB | (*(_BYTE *)(a1 + 18) >> 2) & 4;
  *(_BYTE *)(a2 + 33) = v4;
  *(_BYTE *)(a2 + 33) = (*(_BYTE *)(a1 + 18) >> 2) & 8 | v4 & 0xF7;
  v5 = 1;
  if ( dword_4F04C44 == -1 )
    v5 = (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 6) & 2) != 0;
  v6 = 32 * v5;
  result = (32 * v5) | *(_BYTE *)(a2 + 33) & 0xDFu;
  *(_BYTE *)(a2 + 33) = v6 | *(_BYTE *)(a2 + 33) & 0xDF;
  if ( (*(_BYTE *)(a1 + 16) & 0x20) != 0 )
  {
    result = *(_QWORD *)(a1 + 56);
    if ( result )
      *(_QWORD *)(a2 + 16) = result;
  }
  if ( (*(_BYTE *)(a1 + 18) & 1) != 0 )
  {
    *(_QWORD *)(a2 + 24) = 0;
    result = *(_QWORD *)(a1 + 40);
    v9[0] = (__int64 *)result;
    if ( result )
    {
      v8 = 0;
      if ( *(_BYTE *)(result + 8) != 3 )
        goto LABEL_9;
      sub_72F220(v9);
      result = (__int64)v9[0];
      if ( v9[0] )
      {
        v8 = *(_QWORD *)(a2 + 24);
LABEL_9:
        while ( 1 )
        {
          *(_QWORD *)(a2 + 24) = ++v8;
          result = *(_QWORD *)result;
          v9[0] = (__int64 *)result;
          if ( !result )
            break;
          if ( *(_BYTE *)(result + 8) == 3 )
          {
            sub_72F220(v9);
            result = (__int64)v9[0];
            if ( !v9[0] )
              return result;
            v8 = *(_QWORD *)(a2 + 24);
          }
        }
      }
    }
  }
  return result;
}
