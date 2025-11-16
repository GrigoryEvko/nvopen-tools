// Function: sub_1230940
// Address: 0x1230940
//
__int64 __fastcall sub_1230940(__int64 a1, __int64 *a2, __int64 *a3, int a4, char a5)
{
  unsigned __int64 v7; // r15
  unsigned int v8; // r9d
  __int64 v9; // rdx
  int v10; // eax
  __int64 v11; // rax
  int v13; // ecx
  __int64 v14; // [rsp+18h] [rbp-68h] BYREF
  _QWORD v15[4]; // [rsp+20h] [rbp-60h] BYREF
  __int16 v16; // [rsp+40h] [rbp-40h]

  v7 = *(_QWORD *)(a1 + 232);
  v8 = sub_122FE20((__int64 **)a1, &v14, a3);
  if ( !(_BYTE)v8 )
  {
    v9 = *(_QWORD *)(v14 + 8);
    if ( a5 )
    {
      v10 = *(unsigned __int8 *)(v9 + 8);
      if ( (unsigned int)(v10 - 17) <= 1 )
        LOBYTE(v10) = *(_BYTE *)(**(_QWORD **)(v9 + 16) + 8LL);
      if ( (unsigned __int8)v10 > 3u && (_BYTE)v10 != 5 && (v10 & 0xFD) != 4 )
      {
LABEL_11:
        v16 = 259;
        v15[0] = "invalid operand type for instruction";
        sub_11FD800(a1 + 176, v7, (__int64)v15, 1);
        return 1;
      }
    }
    else
    {
      v13 = *(unsigned __int8 *)(v9 + 8);
      if ( (unsigned int)(v13 - 17) <= 1 )
        LOBYTE(v13) = *(_BYTE *)(**(_QWORD **)(v9 + 16) + 8LL);
      if ( (_BYTE)v13 != 12 )
        goto LABEL_11;
    }
    v16 = 257;
    v11 = sub_B50340(a4, v14, (__int64)v15, 0, 0);
    v8 = 0;
    *a2 = v11;
  }
  return v8;
}
