// Function: sub_35058C0
// Address: 0x35058c0
//
unsigned __int64 __fastcall sub_35058C0(_QWORD *a1, _BYTE *a2)
{
  _QWORD *v2; // rax
  unsigned __int64 result; // rax
  unsigned __int8 v4; // dl
  __int64 v5; // rax
  _BYTE *v6; // [rsp+8h] [rbp-68h] BYREF
  char v7; // [rsp+17h] [rbp-59h] BYREF
  __int64 v8; // [rsp+18h] [rbp-58h] BYREF
  unsigned __int64 *v9; // [rsp+20h] [rbp-50h] BYREF
  __int64 v10; // [rsp+28h] [rbp-48h] BYREF
  _QWORD v11[8]; // [rsp+30h] [rbp-40h] BYREF

  v6 = sub_AF3520(a2);
  v2 = sub_32239E0(a1 + 1, (__int64 *)&v6);
  if ( v2 )
    return (unsigned __int64)(v2 + 2);
  v8 = 0;
  if ( (unsigned int)(unsigned __int8)*v6 - 19 <= 1 )
  {
    v4 = *(v6 - 16);
    if ( (v4 & 2) != 0 )
      v5 = *((_QWORD *)v6 - 4);
    else
      v5 = (__int64)&v6[-8 * ((v4 >> 2) & 0xF) - 16];
    v8 = sub_35057B0(a1, *(unsigned __int8 **)(v5 + 8), 0);
  }
  v7 = 0;
  v11[0] = &v7;
  v11[1] = &v10;
  v10 = 0;
  v11[2] = &v6;
  v11[3] = &v8;
  v9 = (unsigned __int64 *)&v6;
  result = sub_35045A0(a1 + 1, &v9, (__int64)v11) + 16;
  if ( !v8 )
    a1[28] = result;
  return result;
}
