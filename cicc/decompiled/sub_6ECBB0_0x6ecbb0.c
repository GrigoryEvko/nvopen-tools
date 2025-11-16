// Function: sub_6ECBB0
// Address: 0x6ecbb0
//
__int64 __fastcall sub_6ECBB0(__int64 a1, __int64 a2, __int64 *a3)
{
  __int64 v3; // rbp
  char v5; // dl
  __int64 v6; // rax
  __int64 v8[2]; // [rsp-10h] [rbp-10h] BYREF

  v5 = *(_BYTE *)(a1 + 140);
  if ( v5 == 12 )
  {
    v6 = a1;
    do
    {
      v6 = *(_QWORD *)(v6 + 160);
      v5 = *(_BYTE *)(v6 + 140);
    }
    while ( v5 == 12 );
  }
  if ( !v5 )
    return sub_7305B0(a1, a2);
  v8[1] = v3;
  return sub_6ECAE0(a1, 0, 1, 1, a2, a3, v8);
}
