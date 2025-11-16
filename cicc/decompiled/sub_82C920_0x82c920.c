// Function: sub_82C920
// Address: 0x82c920
//
void __fastcall sub_82C920(__int64 a1, __int64 *a2, _QWORD *a3, _QWORD *a4)
{
  __int64 v7; // rdi
  char v8; // al
  __int64 i; // rax
  char v10; // dl
  __int64 v11; // rax
  __int64 *v12; // rdi
  _BYTE v13[80]; // [rsp+0h] [rbp-50h] BYREF

  v7 = *(_QWORD *)(a1 + 136);
  v8 = *(_BYTE *)(v7 + 80);
  if ( v8 == 16 )
  {
    v7 = **(_QWORD **)(v7 + 88);
    v8 = *(_BYTE *)(v7 + 80);
  }
  if ( v8 == 24 )
    v7 = *(_QWORD *)(v7 + 88);
  for ( i = sub_82C1B0(v7, 0, 0, (__int64)v13); i; i = sub_82C230(v13) )
  {
    v10 = *(_BYTE *)(i + 80);
    if ( v10 == 16 )
    {
      i = **(_QWORD **)(i + 88);
      v10 = *(_BYTE *)(i + 80);
    }
    if ( v10 == 24 )
    {
      i = *(_QWORD *)(i + 88);
      v10 = *(_BYTE *)(i + 80);
    }
    v11 = *(_QWORD *)(i + 88);
    if ( v10 == 20 )
      v11 = *(_QWORD *)(v11 + 176);
    sub_7D38C0(*(_QWORD *)(v11 + 152), a2);
  }
  v12 = *(__int64 **)(a1 + 104);
  if ( v12 )
    sub_7D3B00(v12, a2, a3, a4);
}
