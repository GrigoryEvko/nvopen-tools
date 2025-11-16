// Function: sub_3286810
// Address: 0x3286810
//
__int64 __fastcall sub_3286810(__int64 *a1, unsigned int a2, __int64 a3, __int64 a4, __int64 a5, int a6, __int128 a7)
{
  __int64 v10; // rax
  __int16 v11; // dx
  __int64 v12; // rax
  __int64 result; // rax
  __int64 v14; // rax
  __int16 v15; // dx
  __int64 v16; // rax
  __int64 v17; // rcx
  __int128 v18; // [rsp-10h] [rbp-70h]
  __int16 v20; // [rsp+10h] [rbp-50h] BYREF
  __int64 v21; // [rsp+18h] [rbp-48h]
  __int16 v22; // [rsp+20h] [rbp-40h] BYREF
  __int64 v23; // [rsp+28h] [rbp-38h]

  v10 = *(_QWORD *)(a4 + 48) + 16LL * (unsigned int)a5;
  v11 = *(_WORD *)v10;
  v12 = *(_QWORD *)(v10 + 8);
  v20 = v11;
  v21 = v12;
  if ( v11 )
  {
    if ( (unsigned __int16)(v11 - 10) <= 6u
      || (unsigned __int16)(v11 - 126) <= 0x31u
      || (unsigned __int16)(v11 - 208) <= 0x14u )
    {
LABEL_5:
      result = 0;
      if ( (a6 & 0x880) != 0x880 )
        return result;
      goto LABEL_10;
    }
  }
  else if ( (unsigned __int8)sub_3007030((__int64)&v20) )
  {
    goto LABEL_5;
  }
  v14 = *(_QWORD *)(a7 + 48) + 16LL * DWORD2(a7);
  v15 = *(_WORD *)v14;
  v16 = *(_QWORD *)(v14 + 8);
  v22 = v15;
  v23 = v16;
  if ( v15 )
  {
    if ( (unsigned __int16)(v15 - 10) <= 6u
      || (unsigned __int16)(v15 - 126) <= 0x31u
      || (unsigned __int16)(v15 - 208) <= 0x14u )
    {
      goto LABEL_5;
    }
  }
  else if ( (unsigned __int8)sub_3007030((__int64)&v22) )
  {
    goto LABEL_5;
  }
LABEL_10:
  result = sub_3285EA0(a1, a2, a3, a4, a5, a6, a7);
  if ( !result )
  {
    *((_QWORD *)&v18 + 1) = a5;
    *(_QWORD *)&v18 = a4;
    v17 = sub_3285EA0(a1, a2, a3, a7, *((__int64 *)&a7 + 1), a6, v18);
    result = 0;
    if ( v17 )
      return v17;
  }
  return result;
}
