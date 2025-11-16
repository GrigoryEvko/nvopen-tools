// Function: sub_1A1E4F0
// Address: 0x1a1e4f0
//
__int64 __fastcall sub_1A1E4F0(__int64 a1, __int64 a2, __int64 *a3, __int64 a4, __int64 **a5)
{
  __int64 v5; // r13
  __int64 v6; // rbx
  __int64 v7; // rax
  char v8; // al
  unsigned __int64 v9; // r8
  char v11; // al
  __int64 v13; // [rsp+10h] [rbp-D0h]
  __int64 *v14; // [rsp+18h] [rbp-C8h]
  __int64 v15; // [rsp+20h] [rbp-C0h]
  unsigned __int64 v16; // [rsp+28h] [rbp-B8h]
  unsigned __int64 v17; // [rsp+38h] [rbp-A8h]
  __int64 v19; // [rsp+48h] [rbp-98h]
  __int64 v20; // [rsp+50h] [rbp-90h] BYREF
  __int64 v21; // [rsp+58h] [rbp-88h]
  __int64 v22; // [rsp+60h] [rbp-80h]
  __int64 v23; // [rsp+68h] [rbp-78h]
  __int64 v24; // [rsp+70h] [rbp-70h]
  __int64 v25; // [rsp+80h] [rbp-60h] BYREF
  __int64 v26; // [rsp+88h] [rbp-58h]
  __int64 v27; // [rsp+90h] [rbp-50h]
  __int64 v28; // [rsp+98h] [rbp-48h]
  __int64 v29; // [rsp+A0h] [rbp-40h]

  v19 = *(a3 - 3);
  v13 = *a3;
  v5 = a4 + 24;
  v17 = (unsigned __int64)(sub_127FA20(a1, *a3) + 7) >> 3;
  while ( a3 != (__int64 *)(v5 - 24) )
  {
    if ( *(_BYTE *)(v5 - 8) == 55 )
    {
      v6 = *(_QWORD *)(v5 - 48);
      v14 = *(__int64 **)(v5 - 72);
      v15 = *v14;
      v7 = sub_127FA20(a1, *v14);
      v25 = v6;
      v26 = 1;
      v16 = (unsigned __int64)(v7 + 7) >> 3;
      v27 = 0;
      v28 = 0;
      v29 = 0;
      v20 = v19;
      v21 = 1;
      v22 = 0;
      v23 = 0;
      v24 = 0;
      v8 = sub_134CB50(a2, (__int64)&v20, (__int64)&v25);
      v9 = v16;
      if ( v8 == 3 && v17 == v16 && (v11 = sub_1A1E350(a1, v15, v13), v9 = v16, v11) )
      {
        *a5 = v14;
      }
      else
      {
        v25 = v6;
        v20 = v19;
        v26 = v9;
        v27 = 0;
        v28 = 0;
        v29 = 0;
        v21 = v17;
        v22 = 0;
        v23 = 0;
        v24 = 0;
        if ( (unsigned __int8)sub_134CB50(a2, (__int64)&v20, (__int64)&v25) )
          return 0;
      }
    }
    else if ( (unsigned __int8)sub_15F3040(v5 - 24) )
    {
      return 0;
    }
    v5 = *(_QWORD *)(v5 + 8);
    if ( !v5 )
      BUG();
  }
  return 1;
}
