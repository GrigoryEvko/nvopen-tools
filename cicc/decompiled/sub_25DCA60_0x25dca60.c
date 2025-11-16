// Function: sub_25DCA60
// Address: 0x25dca60
//
unsigned __int64 __fastcall sub_25DCA60(__int64 a1, int a2)
{
  __int64 *v2; // r12
  char v3; // r8
  unsigned __int64 result; // rax
  __int64 v5; // r12
  __int64 v6; // r15
  char v7; // r10
  __int64 *v8; // [rsp+10h] [rbp-50h]
  __int64 v9; // [rsp+18h] [rbp-48h]
  int v10; // [rsp+24h] [rbp-3Ch] BYREF
  __int64 v11[7]; // [rsp+28h] [rbp-38h] BYREF

  v11[0] = *(_QWORD *)(a1 + 120);
  v2 = (__int64 *)sub_B2BE50(a1);
  v3 = sub_A74390(v11, a2, &v10);
  result = v11[0];
  if ( v3 )
    result = sub_A7B980(v11, v2, v10, a2);
  v5 = *(_QWORD *)(a1 + 16);
  for ( *(_QWORD *)(a1 + 120) = result; v5; v5 = *(_QWORD *)(v5 + 8) )
  {
    v6 = *(_QWORD *)(v5 + 24);
    if ( *(_BYTE *)v6 != 4 )
    {
      v9 = *(_QWORD *)(v6 + 72);
      v8 = (__int64 *)sub_B2BE50(a1);
      v11[0] = v9;
      v7 = sub_A74390(v11, a2, &v10);
      result = v11[0];
      if ( v7 )
        result = sub_A7B980(v11, v8, v10, a2);
      *(_QWORD *)(v6 + 72) = result;
    }
  }
  return result;
}
