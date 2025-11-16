// Function: sub_16D1F70
// Address: 0x16d1f70
//
__int64 __fastcall sub_16D1F70(_QWORD *a1, __int64 a2, unsigned __int64 a3)
{
  unsigned __int64 v3; // r9
  unsigned __int64 v5; // rbx
  __int64 v6; // rcx
  unsigned __int8 v7; // dl
  unsigned __int8 v8; // al
  __int64 result; // rax

  v3 = a3;
  v5 = a1[1];
  if ( v5 <= a3 )
    v3 = a1[1];
  if ( v3 )
  {
    v6 = 0;
    while ( 1 )
    {
      v7 = *(_BYTE *)(*a1 + v6);
      if ( (unsigned __int8)(v7 - 65) < 0x1Au )
        v7 += 32;
      v8 = *(_BYTE *)(a2 + v6);
      if ( (unsigned __int8)(v8 - 65) < 0x1Au )
        v8 += 32;
      if ( v8 != v7 )
        return v7 < v8 ? -1 : 1;
      if ( v3 == ++v6 )
        goto LABEL_12;
    }
  }
  else
  {
LABEL_12:
    result = 0;
    if ( v5 != a3 )
      return v5 < a3 ? -1 : 1;
  }
  return result;
}
