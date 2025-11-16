// Function: sub_1568EC0
// Address: 0x1568ec0
//
__int64 __fastcall sub_1568EC0(__int64 a1, __int64 a2)
{
  __int64 v2; // rdx
  __int64 result; // rax
  _QWORD *v4; // r12
  __int64 v5; // rax
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // rax
  __int64 v15; // rdx
  _QWORD v16[2]; // [rsp+0h] [rbp-40h] BYREF
  __int64 v17; // [rsp+10h] [rbp-30h] BYREF
  __int64 v18; // [rsp+18h] [rbp-28h]
  __int64 v19; // [rsp+20h] [rbp-20h]
  __int64 v20; // [rsp+28h] [rbp-18h]

  v2 = *(unsigned int *)(a1 + 8);
  if ( (unsigned __int8)(**(_BYTE **)(a1 - 8 * v2) - 4) > 0x1Eu || (result = a1, (unsigned int)v2 <= 2) )
  {
    v4 = (_QWORD *)(*(_QWORD *)(a1 + 16) & 0xFFFFFFFFFFFFFFF8LL);
    if ( (*(_QWORD *)(a1 + 16) & 4) != 0 )
    {
      v4 = (_QWORD *)*v4;
      if ( (_DWORD)v2 != 3 )
        goto LABEL_5;
    }
    else if ( (_DWORD)v2 != 3 )
    {
LABEL_5:
      v17 = a1;
      v18 = a1;
      v5 = sub_1643360(v4);
      v6 = sub_15A06D0(v5);
      v19 = sub_1624210(v6, a2, v7, v8);
      return sub_1627350(v4, &v17, 3, 0, 1);
    }
    v9 = *(_QWORD *)(a1 - 16);
    v16[0] = *(_QWORD *)(a1 - 8 * v2);
    v16[1] = v9;
    v17 = sub_1627350(v4, v16, 2, 0, 1);
    v18 = v17;
    v10 = sub_1643360(v4);
    v11 = sub_15A06D0(v10);
    v14 = sub_1624210(v11, v16, v12, v13);
    v15 = *(unsigned int *)(a1 + 8);
    v19 = v14;
    v20 = *(_QWORD *)(a1 + 8 * (2 - v15));
    return sub_1627350(v4, &v17, 4, 0, 1);
  }
  return result;
}
