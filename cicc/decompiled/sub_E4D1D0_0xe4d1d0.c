// Function: sub_E4D1D0
// Address: 0xe4d1d0
//
__int64 __fastcall sub_E4D1D0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, unsigned int a5)
{
  __int64 *v8; // rax
  __int64 v9; // rdx
  __int64 *v10; // rax
  bool v11; // zf
  __int64 v12; // rax
  __int64 v13; // rax
  char v14; // dl
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v18; // r8
  __int64 v19; // r9
  __int64 v20; // rax
  __int64 v21; // r8
  __int64 v22; // r9
  __int64 v23; // rax
  __int64 v24; // r8
  __int64 v25; // r9
  __int64 v26; // rax
  __int64 v27; // r8
  __int64 v28; // r9
  __int64 v29; // rax
  __int64 v30; // [rsp+8h] [rbp-68h] BYREF
  _QWORD v31[2]; // [rsp+10h] [rbp-60h] BYREF
  __int64 *v32; // [rsp+20h] [rbp-50h]
  __int64 v33; // [rsp+28h] [rbp-48h]
  __int16 v34; // [rsp+30h] [rbp-40h]

  v30 = a2;
  if ( (*(_BYTE *)(a4 + 8) & 1) != 0 )
  {
    v8 = *(__int64 **)(a4 - 8);
    v9 = *v8;
    v10 = v8 + 3;
  }
  else
  {
    v9 = 0;
    v10 = 0;
  }
  v32 = v10;
  v11 = *(_BYTE *)(a1 + 745) == 0;
  v34 = 1283;
  v31[0] = "Set address to ";
  v33 = v9;
  if ( !v11 )
  {
    sub_CA0EC0((__int64)v31, a1 + 488);
    v23 = *(_QWORD *)(a1 + 496);
    if ( (unsigned __int64)(v23 + 1) > *(_QWORD *)(a1 + 504) )
    {
      sub_C8D290(a1 + 488, (const void *)(a1 + 512), v23 + 1, 1u, v21, v22);
      v23 = *(_QWORD *)(a1 + 496);
    }
    *(_BYTE *)(*(_QWORD *)(a1 + 488) + v23) = 10;
    ++*(_QWORD *)(a1 + 496);
  }
  v12 = sub_E81A90(0, *(_QWORD *)(a1 + 8), 0, 0);
  sub_E9A5B0(a1, v12, 1, 0);
  sub_E98EB0(a1, a5 + 1, 0);
  v13 = sub_E81A90(2, *(_QWORD *)(a1 + 8), 0, 0);
  sub_E9A5B0(a1, v13, 1, 0);
  sub_E9A500(a1, a4, a5, 0);
  if ( a3 )
  {
    v14 = *(_BYTE *)(a1 + 745);
    if ( v30 == 0x7FFFFFFFFFFFFFFFLL )
    {
      v31[0] = "End sequence";
      v34 = 259;
      if ( v14 )
      {
        sub_CA0EC0((__int64)v31, a1 + 488);
        v26 = *(_QWORD *)(a1 + 496);
        if ( (unsigned __int64)(v26 + 1) > *(_QWORD *)(a1 + 504) )
        {
          sub_C8D290(a1 + 488, (const void *)(a1 + 512), v26 + 1, 1u, v24, v25);
          v26 = *(_QWORD *)(a1 + 496);
        }
        *(_BYTE *)(*(_QWORD *)(a1 + 488) + v26) = 10;
        ++*(_QWORD *)(a1 + 496);
      }
      sub_E4CAB0(a1, 0, 1u);
      sub_E98EB0(a1, 1, 0);
      return sub_E4CAB0(a1, 1, 1u);
    }
    else
    {
      v31[0] = "Advance line ";
      v32 = &v30;
      v34 = 3075;
      if ( v14 )
      {
        sub_CA0EC0((__int64)v31, a1 + 488);
        v20 = *(_QWORD *)(a1 + 496);
        if ( (unsigned __int64)(v20 + 1) > *(_QWORD *)(a1 + 504) )
        {
          sub_C8D290(a1 + 488, (const void *)(a1 + 512), v20 + 1, 1u, v18, v19);
          v20 = *(_QWORD *)(a1 + 496);
        }
        *(_BYTE *)(*(_QWORD *)(a1 + 488) + v20) = 10;
        ++*(_QWORD *)(a1 + 496);
      }
      v15 = sub_E81A90(3, *(_QWORD *)(a1 + 8), 0, 0);
      sub_E9A5B0(a1, v15, 1, 0);
      sub_E990E0(a1, v30);
      v16 = sub_E81A90(1, *(_QWORD *)(a1 + 8), 0, 0);
      return sub_E9A5B0(a1, v16, 1, 0);
    }
  }
  else
  {
    v11 = *(_BYTE *)(a1 + 745) == 0;
    v31[0] = "Start sequence";
    v34 = 259;
    if ( !v11 )
    {
      sub_CA0EC0((__int64)v31, a1 + 488);
      v29 = *(_QWORD *)(a1 + 496);
      if ( (unsigned __int64)(v29 + 1) > *(_QWORD *)(a1 + 504) )
      {
        sub_C8D290(a1 + 488, (const void *)(a1 + 512), v29 + 1, 1u, v27, v28);
        v29 = *(_QWORD *)(a1 + 496);
      }
      *(_BYTE *)(*(_QWORD *)(a1 + 488) + v29) = 10;
      ++*(_QWORD *)(a1 + 496);
    }
    BYTE2(v31[0]) = 14;
    LOWORD(v31[0]) = -1267;
    return sub_E77F70(a1, v31[0], v30, 0);
  }
}
