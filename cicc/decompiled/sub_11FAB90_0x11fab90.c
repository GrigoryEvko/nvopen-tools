// Function: sub_11FAB90
// Address: 0x11fab90
//
__int64 __fastcall sub_11FAB90(__int64 a1, char a2, __int64 a3, __int64 a4, _BYTE *a5)
{
  __int64 v5; // r9
  __int64 v8; // r8
  __int64 v9; // rax
  bool v10; // zf
  __int64 result; // rax
  const char *v12; // rax
  __int64 v13; // rdx
  void *v14; // rdx
  __int64 v15; // [rsp+8h] [rbp-D8h]
  __int64 *v17; // [rsp+18h] [rbp-C8h] BYREF
  void *v18[4]; // [rsp+20h] [rbp-C0h] BYREF
  __int16 v19; // [rsp+40h] [rbp-A0h]
  void *v20; // [rsp+50h] [rbp-90h] BYREF
  __int16 v21; // [rsp+70h] [rbp-70h]
  __int64 v22[12]; // [rsp+80h] [rbp-60h] BYREF

  v5 = a4;
  if ( !qword_4F92310
    || (v12 = sub_BD5D20(a1),
        v22[1] = v13,
        v22[0] = (__int64)v12,
        result = sub_C931B0(v22, (_WORD *)qword_4F92308, qword_4F92310, 0),
        v5 = a4,
        result != -1) )
  {
    v8 = 0;
    if ( a3 )
    {
      v15 = v5;
      v9 = sub_11FCC10(a1, a3);
      v5 = v15;
      v8 = v9;
    }
    sub_11F3840((__int64)v22, a1, a3, v5, v8);
    v21 = 257;
    if ( a5 )
    {
      v10 = *a5 == 0;
      v19 = 257;
      if ( !v10 )
      {
        v18[0] = a5;
        LOBYTE(v19) = 3;
      }
    }
    else
    {
      v18[0] = "cfg";
      v19 = 1283;
      v18[2] = (void *)sub_BD5D20(a1);
      v18[3] = v14;
    }
    v17 = v22;
    sub_11FA7D0(&v17, v18, a2, &v20, 0);
    return sub_11F3870((__int64)v22);
  }
  return result;
}
