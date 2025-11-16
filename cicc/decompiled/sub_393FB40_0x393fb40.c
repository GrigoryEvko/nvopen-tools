// Function: sub_393FB40
// Address: 0x393fb40
//
__int64 __fastcall sub_393FB40(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // r8
  __int64 v5; // r9
  char v6; // al
  __int64 v8; // rdx
  _QWORD *v9; // r14
  unsigned __int64 v10; // rdx
  _QWORD *v11; // [rsp+0h] [rbp-60h] BYREF
  __int64 v12; // [rsp+8h] [rbp-58h]
  char v13; // [rsp+10h] [rbp-50h]
  _QWORD v14[2]; // [rsp+20h] [rbp-40h] BYREF
  char v15; // [rsp+30h] [rbp-30h]

  sub_16C2E90((__int64)v14, a2, 0xFFFFFFFFFFFFFFFFLL, 1);
  v6 = v15 & 1;
  if ( (v15 & 1) != 0 && LODWORD(v14[0]) )
  {
    v13 |= 1u;
    LODWORD(v11) = v14[0];
    v12 = v14[1];
  }
  else
  {
    v9 = (_QWORD *)v14[0];
    v14[0] = 0;
    v10 = v9[2] - v9[1];
    if ( v10 > 0xFFFFFFFF )
    {
      v13 |= 1u;
      LODWORD(v11) = 3;
      v12 = sub_393D180((__int64)v14, a2, v10, 0xFFFFFFFFLL, v4, v5);
      (*(void (__fastcall **)(_QWORD *))(*v9 + 8LL))(v9);
      v6 = v15 & 1;
    }
    else
    {
      v13 &= ~1u;
      v11 = v9;
    }
    if ( !v6 && v14[0] )
      (*(void (__fastcall **)(_QWORD))(*(_QWORD *)v14[0] + 8LL))(v14[0]);
  }
  if ( (v13 & 1) != 0 && (v8 = v12, (_DWORD)v11) )
  {
    *(_DWORD *)a1 = (_DWORD)v11;
    *(_BYTE *)(a1 + 16) |= 1u;
    *(_QWORD *)(a1 + 8) = v8;
    return a1;
  }
  else
  {
    sub_393F820(a1, (__int64 *)&v11, a3);
    if ( (v13 & 1) == 0 )
    {
      if ( v11 )
        (*(void (__fastcall **)(_QWORD *))(*v11 + 8LL))(v11);
    }
    return a1;
  }
}
