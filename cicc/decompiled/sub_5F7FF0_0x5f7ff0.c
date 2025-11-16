// Function: sub_5F7FF0
// Address: 0x5f7ff0
//
void __fastcall sub_5F7FF0(__int64 a1)
{
  __int64 v1; // rbx
  __int64 v2; // rax
  char v3; // dl
  _DWORD v4[3]; // [rsp+Ch] [rbp-14h] BYREF

  v1 = *(_QWORD *)(a1 + 168);
  if ( !*(_QWORD *)(v1 + 184) )
  {
    v2 = sub_5F7F60(a1, v4);
    if ( v2 )
    {
      if ( !v4[0] )
      {
        v3 = *(_BYTE *)(v2 + 80);
        if ( v3 == 16 )
        {
          v2 = **(_QWORD **)(v2 + 88);
          v3 = *(_BYTE *)(v2 + 80);
        }
        if ( v3 == 24 )
          v2 = *(_QWORD *)(v2 + 88);
        *(_QWORD *)(v1 + 184) = *(_QWORD *)(v2 + 88);
      }
    }
  }
}
