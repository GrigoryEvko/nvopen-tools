// Function: sub_11A7520
// Address: 0x11a7520
//
__int64 __fastcall sub_11A7520(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 *a6, __int64 a7)
{
  int v10; // eax
  bool v11; // al
  int v12; // eax
  bool v13; // al
  __int64 *v15; // [rsp+8h] [rbp-48h]
  __int64 *v16; // [rsp+8h] [rbp-48h]
  int v17; // [rsp+14h] [rbp-3Ch]
  int v18; // [rsp+14h] [rbp-3Ch]

  if ( *(_DWORD *)(a5 + 8) <= 0x40u )
  {
    v11 = *(_QWORD *)a5 == 0;
  }
  else
  {
    v15 = a6;
    v17 = *(_DWORD *)(a5 + 8);
    v10 = sub_C444A0(a5);
    a6 = v15;
    v11 = v17 == v10;
  }
  if ( v11 )
    return 0;
  if ( *(_DWORD *)(a3 + 8) <= 0x40u )
  {
    v13 = *(_QWORD *)a3 == 0;
  }
  else
  {
    v16 = a6;
    v18 = *(_DWORD *)(a3 + 8);
    v12 = sub_C444A0(a3);
    a6 = v16;
    v13 = v18 == v12;
  }
  if ( v13 )
    return 0;
  else
    return sub_11A6910(a1, a2, a3, a4, a5, a6, a7);
}
