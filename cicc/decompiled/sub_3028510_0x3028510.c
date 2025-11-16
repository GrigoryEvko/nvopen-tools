// Function: sub_3028510
// Address: 0x3028510
//
void __fastcall sub_3028510(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  int v6; // r8d
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 v9; // r15
  __int64 v10; // rax
  __int64 v11; // r8
  char v12; // al
  __int64 v13; // rax
  __int64 v14; // r15
  __int64 v15; // rbx
  __int64 v16; // rax
  __int64 v17; // [rsp+0h] [rbp-F0h]
  int v18; // [rsp+Ch] [rbp-E4h]
  int v19; // [rsp+Ch] [rbp-E4h]
  __int64 i; // [rsp+18h] [rbp-D8h]
  __int64 v23; // [rsp+20h] [rbp-D0h] BYREF
  __int64 v24; // [rsp+28h] [rbp-C8h]
  _BYTE v25[24]; // [rsp+30h] [rbp-C0h] BYREF
  char *v26; // [rsp+48h] [rbp-A8h]
  char v27; // [rsp+58h] [rbp-98h] BYREF
  char v28; // [rsp+94h] [rbp-5Ch]
  __int64 v29; // [rsp+A0h] [rbp-50h]

  v6 = *(_DWORD *)(a5 + 4);
  if ( v6 )
  {
    if ( v6 >= 0 )
    {
      v17 = 0;
    }
    else
    {
      v18 = v6;
      v7 = (*(__int64 (__fastcall **)(__int64 *, _QWORD, _QWORD))(*a1 + 480))(a1, (unsigned int)v6, 0);
      v6 = v18;
      v17 = v7;
      if ( !v7 )
        return;
    }
    v19 = v6;
    v8 = sub_A777F0(0x10u, a1 + 123);
    v9 = v8;
    if ( v8 )
    {
      *(_QWORD *)v8 = 0;
      *(_DWORD *)(v8 + 8) = 0;
    }
    sub_3247620(v25, a1, a2, v8);
    if ( !*(_BYTE *)a5 )
      v28 = v28 & 0xF8 | 2;
    if ( v19 < 0 )
    {
      LODWORD(v23) = 65547;
      sub_3249A20(a2, v9, 0, v23, 146);
      LODWORD(v23) = 65551;
      sub_3249A20(a2, v9, 0, v23, v17);
      LODWORD(v23) = 65549;
      sub_3249A20(a2, v9, 0, v23, 0);
      v11 = 12;
    }
    else
    {
      v10 = (*(__int64 (__fastcall **)(__int64 *))(*a1 + 184))(a1);
      sub_324BB60(a2, v9, v10);
      v11 = 6;
    }
    LODWORD(v23) = 65547;
    sub_3249A20(a2, a4 + 8, 51, v23, v11);
    v12 = *(_BYTE *)(a3 + 88);
    if ( v12 == 3 )
    {
      v13 = sub_321E240(a3 + 40);
      v14 = *(_QWORD *)(v13 + 24);
      for ( i = v13 + 8; i != v14; v14 = sub_220EF30(v14) )
      {
        v15 = *(_QWORD *)(v14 + 40);
        sub_3243D60(v25, v15);
        v23 = 0;
        v24 = 0;
        if ( v15 )
        {
          v23 = *(_QWORD *)(v15 + 16);
          v24 = *(_QWORD *)(v15 + 24);
        }
        sub_3244870(v25, &v23);
      }
    }
    else if ( v12 == 1 )
    {
      v23 = 0;
      v24 = 0;
      v16 = *(_QWORD *)(a3 + 48);
      if ( v16 )
      {
        v23 = *(_QWORD *)(v16 + 16);
        v24 = *(_QWORD *)(v16 + 24);
      }
      sub_3244870(v25, &v23);
    }
    sub_3243D40(v25);
    sub_3249620(a2, a4, 2, v29);
    if ( v26 != &v27 )
      _libc_free((unsigned __int64)v26);
  }
}
