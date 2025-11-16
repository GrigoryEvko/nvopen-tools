// Function: sub_326C2C0
// Address: 0x326c2c0
//
__int64 __fastcall sub_326C2C0(__int64 a1, __int64 a2, __int64 a3, char a4)
{
  int v7; // eax
  __int64 result; // rax
  __int64 v9; // rsi
  __int64 v10; // r9
  unsigned int v11; // ebx
  unsigned __int16 v12; // ax
  __int64 v13; // rsi
  __int64 v14; // r9
  unsigned int v15; // ebx
  __int64 v16; // [rsp+8h] [rbp-48h]
  __int64 v17; // [rsp+8h] [rbp-48h]
  __int64 v18; // [rsp+8h] [rbp-48h]
  __int64 v19; // [rsp+10h] [rbp-40h] BYREF
  int v20; // [rsp+18h] [rbp-38h]

  v7 = *(_DWORD *)(a1 + 24);
  if ( a4 && (v7 == 11 || v7 == 35) )
  {
    v9 = *(_QWORD *)(a1 + 80);
    v10 = *(_QWORD *)(*(_QWORD *)(a1 + 48) + 16LL * (unsigned int)a2 + 8);
    v11 = *(unsigned __int16 *)(*(_QWORD *)(a1 + 48) + 16LL * (unsigned int)a2);
    v19 = v9;
    if ( v9 )
    {
      v16 = v10;
      sub_B96E90((__int64)&v19, v9, 1);
      v10 = v16;
    }
    v20 = *(_DWORD *)(a1 + 72);
    result = sub_3407510(a3, &v19, a1, a2, v11, v10);
    if ( v19 )
      goto LABEL_10;
  }
  else
  {
    if ( v7 != 188 )
      return 0;
    v12 = sub_33E2690(a3, *(_QWORD *)(*(_QWORD *)(a1 + 40) + 40LL), *(_QWORD *)(*(_QWORD *)(a1 + 40) + 48LL), 0);
    if ( HIBYTE(v12) && (_BYTE)v12 )
      return **(_QWORD **)(a1 + 40);
    if ( a4 && sub_33DFBC0(*(_QWORD *)(*(_QWORD *)(a1 + 40) + 40LL), *(_QWORD *)(*(_QWORD *)(a1 + 40) + 48LL), 0, 0) )
    {
      v13 = *(_QWORD *)(a1 + 80);
      v14 = *(_QWORD *)(*(_QWORD *)(a1 + 48) + 16LL * (unsigned int)a2 + 8);
      v15 = *(unsigned __int16 *)(*(_QWORD *)(a1 + 48) + 16LL * (unsigned int)a2);
      v19 = v13;
      if ( v13 )
      {
        v18 = v14;
        sub_B96E90((__int64)&v19, v13, 1);
        v14 = v18;
      }
      v20 = *(_DWORD *)(a1 + 72);
      result = sub_3407510(a3, &v19, a1, a2, v15, v14);
      if ( v19 )
      {
LABEL_10:
        v17 = result;
        sub_B91220((__int64)&v19, v19);
        return v17;
      }
    }
    else
    {
      return 0;
    }
  }
  return result;
}
