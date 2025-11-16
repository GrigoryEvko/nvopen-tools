// Function: sub_1208110
// Address: 0x1208110
//
__int64 __fastcall sub_1208110(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned int v4; // r12d
  unsigned __int64 v5; // rsi
  unsigned int v7; // r13d
  unsigned __int64 v8; // r14
  int v9; // eax
  unsigned __int64 v10; // rsi
  unsigned __int64 v11; // rax
  __int64 v12; // [rsp+8h] [rbp-C8h]
  __int64 v13; // [rsp+10h] [rbp-C0h]
  _QWORD v14[4]; // [rsp+20h] [rbp-B0h] BYREF
  __int16 v15; // [rsp+40h] [rbp-90h]
  _QWORD v16[4]; // [rsp+50h] [rbp-80h] BYREF
  __int16 v17; // [rsp+70h] [rbp-60h]
  _QWORD v18[4]; // [rsp+80h] [rbp-50h] BYREF
  __int16 v19; // [rsp+A0h] [rbp-30h]

  if ( *(_DWORD *)(a1 + 240) == 529 )
  {
    v4 = *(unsigned __int8 *)(a1 + 332);
    if ( (_BYTE)v4 )
    {
      v7 = *(_DWORD *)(a1 + 328);
      v8 = *(_QWORD *)(a4 + 16);
      if ( v7 <= 0x40 )
      {
        v11 = *(_QWORD *)(a1 + 320);
        if ( v8 < v11 )
        {
LABEL_7:
          v14[2] = a2;
          v10 = *(_QWORD *)(a1 + 232);
          v15 = 1283;
          v14[0] = "value for '";
          v14[3] = a3;
          v16[0] = v14;
          v18[2] = a4 + 16;
          v16[2] = "' too large, limit is ";
          v17 = 770;
          v19 = 2818;
          v18[0] = v16;
          sub_11FD800(a1 + 176, v10, (__int64)v18, 1);
          return v4;
        }
      }
      else
      {
        v12 = a4;
        v13 = a3;
        v9 = sub_C444A0(a1 + 320);
        a3 = v13;
        a4 = v12;
        if ( v7 - v9 > 0x40 )
          goto LABEL_7;
        v11 = **(_QWORD **)(a1 + 320);
        if ( v8 < v11 )
          goto LABEL_7;
      }
      *(_BYTE *)(a4 + 8) = 1;
      v4 = 0;
      *(_QWORD *)a4 = v11;
      *(_DWORD *)(a1 + 240) = sub_1205200(a1 + 176);
      return v4;
    }
  }
  v5 = *(_QWORD *)(a1 + 232);
  v18[0] = "expected unsigned integer";
  v4 = 1;
  v19 = 259;
  sub_11FD800(a1 + 176, v5, (__int64)v18, 1);
  return v4;
}
