// Function: sub_127C7B0
// Address: 0x127c7b0
//
bool __fastcall sub_127C7B0(__int64 a1, __int64 *a2)
{
  __int64 v2; // rbp
  bool result; // al
  __int64 v4; // rdi
  int v5; // [rsp-1Ch] [rbp-1Ch] BYREF
  __int64 v6; // [rsp-8h] [rbp-8h]

  result = 0;
  if ( *(_BYTE *)(a1 + 24) == 2 )
  {
    v4 = *(_QWORD *)(a1 + 56);
    if ( *(_BYTE *)(v4 + 173) == 1 )
    {
      v6 = v2;
      *a2 = sub_620FD0(v4, &v5);
      return v5 == 0;
    }
  }
  return result;
}
