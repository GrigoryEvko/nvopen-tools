// Function: sub_7D4A40
// Address: 0x7d4a40
//
__int64 __fastcall sub_7D4A40(__int64 *a1, __int64 a2, unsigned int a3, __int64 a4, __int64 a5)
{
  __int64 result; // rax
  char v6; // dl
  int v7; // [rsp+4h] [rbp-1Ch] BYREF
  __int64 v8; // [rsp+8h] [rbp-18h] BYREF

  result = a1[3];
  v8 = 0;
  v7 = 0;
  if ( result
    || (dword_4D042AC && qword_4D049B8[11] == a2
      ? (result = sub_7D4600(unk_4F07288, a1, a3, a4, a5))
      : (result = sub_7D3C60(a1, a2, a3, a2, (int)&v8, 0, &v7), a1[3] = result),
        result) )
  {
    v6 = *(_BYTE *)(result + 80);
    if ( v6 == 16 )
    {
      result = **(_QWORD **)(result + 88);
      if ( *(_BYTE *)(result + 80) != 24 )
        return result;
    }
    else if ( v6 != 24 )
    {
      return result;
    }
    return *(_QWORD *)(result + 88);
  }
  return result;
}
