// Function: sub_1AC9680
// Address: 0x1ac9680
//
_BOOL8 __fastcall sub_1AC9680(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v6; // r14
  __int64 v7; // rbx
  __int64 v9; // rax
  __int64 v10; // r13
  __int64 v11; // r12
  __int64 v12; // rsi
  __int64 v14; // r15
  __int64 v15; // r13
  __int64 *v16; // r12
  __int64 v17; // rdi
  __int64 v18; // rdx
  __int64 v19; // r14
  __int64 v20; // [rsp+8h] [rbp-E8h]
  __int64 v21; // [rsp+10h] [rbp-E0h]
  unsigned __int64 v22; // [rsp+18h] [rbp-D8h]
  __int64 v23; // [rsp+20h] [rbp-D0h]
  __int64 v24; // [rsp+28h] [rbp-C8h]
  unsigned __int64 v25; // [rsp+30h] [rbp-C0h]
  unsigned __int64 v27; // [rsp+40h] [rbp-B0h]
  bool v28; // [rsp+4Fh] [rbp-A1h]
  unsigned __int64 v29; // [rsp+50h] [rbp-A0h]
  __int64 v30; // [rsp+58h] [rbp-98h]
  _QWORD v31[6]; // [rsp+60h] [rbp-90h] BYREF
  _QWORD v32[12]; // [rsp+90h] [rbp-60h] BYREF

  v6 = *(_QWORD *)(a3 + 48);
  if ( v6 )
    v6 -= 24;
  v28 = a3 == a5 || a4 == a2;
  if ( v28 )
  {
    return (a4 == a2) == (a3 == a5);
  }
  else
  {
    v7 = *(_QWORD *)(a4 + 48);
    v27 = sub_157EBA0(a4) + 24;
    v30 = *(_QWORD *)(a5 + 48);
    v22 = sub_157EBA0(a5);
    if ( v27 == v7 )
    {
      return v30 == v22 + 24;
    }
    else
    {
      v24 = v6;
      v25 = sub_157EBA0(a3);
      v9 = 0;
      v10 = v7;
      if ( v6 )
        v9 = v6 + 24;
      v21 = v6 + 24;
      v23 = v9;
      while ( 1 )
      {
        v11 = v10 - 24;
        if ( !v10 )
          v11 = 0;
        v12 = v30 - 24;
        if ( !v30 )
          v12 = 0;
        if ( !sub_15F41F0(v11, v12)
          || ((unsigned __int8)sub_15F3040(v11) || sub_15F3330(v11))
          && (*(_BYTE *)(v11 + 16) != 55 || (*(_BYTE *)(v11 + 18) & 1) != 0)
          || (unsigned __int8)sub_15F2ED0(v11) )
        {
          break;
        }
        if ( (unsigned __int8)sub_15F3040(v11) )
        {
          if ( v24 )
          {
            if ( !v25 )
            {
              v20 = v10;
              v29 = 0;
              v15 = v21;
              v14 = v11;
              v16 = a1;
              goto LABEL_31;
            }
            v29 = v25 + 24;
            if ( v25 + 24 != v23 )
            {
              v20 = v10;
              v14 = v11;
              v15 = v23;
              v16 = a1;
              goto LABEL_31;
            }
          }
          else if ( v25 )
          {
            v20 = v10;
            v29 = v25 + 24;
            v15 = v23;
            v14 = v11;
            v16 = a1;
            while ( 1 )
            {
LABEL_31:
              v18 = v15 - 24;
              if ( !v15 )
                v18 = 0;
              v19 = v18;
              if ( (unsigned __int8)sub_15F2ED0(v18) || (unsigned __int8)sub_15F3040(v19) )
              {
                v17 = *v16;
                if ( !*v16 )
                  return v28;
                v32[0] = v19;
                v32[1] = -1;
                memset(&v32[2], 0, 24);
                v31[0] = v14;
                v31[1] = -1;
                memset(&v31[2], 0, 24);
                if ( (unsigned __int8)sub_134CB50(v17, (__int64)v31, (__int64)v32) )
                  return v28;
              }
              v15 = *(_QWORD *)(v15 + 8);
              if ( v29 == v15 )
              {
                v10 = v20;
                break;
              }
            }
          }
        }
        v10 = *(_QWORD *)(v10 + 8);
        v30 = *(_QWORD *)(v30 + 8);
        if ( v27 == v10 )
          return v30 == v22 + 24;
      }
    }
  }
  return v28;
}
