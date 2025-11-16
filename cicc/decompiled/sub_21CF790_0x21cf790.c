// Function: sub_21CF790
// Address: 0x21cf790
//
__int64 __fastcall sub_21CF790(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, unsigned int a6, __int64 a7)
{
  unsigned __int64 v8; // rbx
  __int64 *v9; // rax
  _DWORD *v11; // rdx
  __int64 v12; // r12
  char v14; // al
  char v15; // al
  bool v16; // al
  _DWORD *v17; // [rsp+8h] [rbp-48h]
  _DWORD v18[13]; // [rsp+1Ch] [rbp-34h] BYREF

  v8 = a4 & 0xFFFFFFFFFFFFFFF8LL;
  if ( (a4 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
    return sub_15A9FE0(a7, a5);
  v9 = (__int64 *)(v8 - 72);
  if ( (a4 & 4) != 0 )
    v9 = (__int64 *)(v8 - 24);
  v18[0] = 0;
  v11 = v18;
  v12 = *v9;
  if ( *(_BYTE *)(*v9 + 16) )
  {
    if ( *(_BYTE *)(v8 + 16) != 78 )
      return sub_15A9FE0(a7, a5);
    v14 = sub_1C30010(v8, a6, v18);
    v11 = v18;
    if ( v14 )
      return v18[0];
    v12 = *(_QWORD *)(v8 - 24);
    v15 = *(_BYTE *)(v12 + 16);
    if ( v15 == 5 )
    {
      while ( 1 )
      {
        v17 = v11;
        v16 = sub_1594510(v12);
        v11 = v17;
        if ( !v16 )
          break;
        v12 = *(_QWORD *)(v12 - 24LL * (*(_DWORD *)(v12 + 20) & 0xFFFFFFF));
        v15 = *(_BYTE *)(v12 + 16);
        if ( v15 != 5 )
          goto LABEL_14;
      }
      v15 = *(_BYTE *)(v12 + 16);
    }
LABEL_14:
    if ( v15 )
      return sub_15A9FE0(a7, a5);
  }
  if ( !(unsigned __int8)sub_1C2FF50(v12, a6, v11) )
    return sub_15A9FE0(a7, a5);
  return v18[0];
}
