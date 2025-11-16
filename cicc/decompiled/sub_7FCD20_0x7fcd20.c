// Function: sub_7FCD20
// Address: 0x7fcd20
//
__int64 *__fastcall sub_7FCD20(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 **a6)
{
  _QWORD v7[3]; // [rsp+0h] [rbp-A0h] BYREF
  int v8; // [rsp+18h] [rbp-88h]
  __int64 v9; // [rsp+20h] [rbp-80h]
  __int64 v10; // [rsp+28h] [rbp-78h]
  __int64 v11; // [rsp+30h] [rbp-70h]
  __int64 v12; // [rsp+38h] [rbp-68h]
  __int64 v13; // [rsp+40h] [rbp-60h]
  __int64 v14; // [rsp+48h] [rbp-58h]
  __int64 v15; // [rsp+50h] [rbp-50h]
  __int64 v16; // [rsp+58h] [rbp-48h]
  __int64 v17; // [rsp+60h] [rbp-40h]
  __int64 v18; // [rsp+68h] [rbp-38h]
  __int64 v19; // [rsp+70h] [rbp-30h]
  __int64 v20; // [rsp+78h] [rbp-28h]
  int v21; // [rsp+80h] [rbp-20h]
  char v22; // [rsp+84h] [rbp-1Ch]
  __int64 v23; // [rsp+88h] [rbp-18h]
  __int64 v24; // [rsp+90h] [rbp-10h]
  __int64 v25; // [rsp+98h] [rbp-8h]

  if ( !a1 )
  {
    v22 = 0;
    memset(v7, 0, sizeof(v7));
    v18 = unk_4D03EC0;
    a1 = v7;
    v8 = 0;
    v9 = 0;
    v10 = 0;
    v11 = 0;
    v12 = 0;
    v13 = 0;
    v14 = -1;
    v15 = 0;
    v16 = 0;
    v17 = 0;
    v19 = 0;
    v20 = 0;
    v21 = 0;
    v23 = 0;
    v24 = 0;
    v25 = 0;
  }
  *a6 = 0;
  if ( (*(_BYTE *)(*(_QWORD *)(a2 + 40) + 176LL) & 0x10) != 0 )
  {
    a1[18] = a4;
    *(_WORD *)((char *)a1 + 129) = 1;
    a1[19] = a2;
  }
  return sub_7FCCD0((__int64)a1, a3, a5, a6);
}
