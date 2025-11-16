// Function: sub_31CD500
// Address: 0x31cd500
//
char __fastcall sub_31CD500(__int64 a1, unsigned __int8 *a2)
{
  __int64 v2; // rax
  __int64 *v3; // r13
  __int64 v4; // r14
  __int64 v5; // rax
  __int64 v6; // rax
  __int64 v7; // rcx
  __int64 v8; // rbx
  __int64 v9; // rax
  _QWORD *v10; // r13
  __int64 v12; // [rsp+8h] [rbp-88h]
  __int64 v13[4]; // [rsp+10h] [rbp-80h] BYREF
  _QWORD v14[4]; // [rsp+30h] [rbp-60h] BYREF
  __int16 v15; // [rsp+50h] [rbp-40h]

  LODWORD(v2) = sub_BD2910(*(_QWORD *)(a1 + 312));
  if ( (_DWORD)v2 == 1 )
  {
    LOBYTE(v2) = sub_B46500(a2);
    if ( !(_BYTE)v2 && (a2[2] & 1) == 0 )
    {
      v2 = *(_QWORD *)(*((_QWORD *)a2 - 4) + 8LL);
      if ( *(_DWORD *)(v2 + 8) <= 0x1FFu )
      {
        v3 = (__int64 *)sub_B43CA0((__int64)a2);
        v12 = *((_QWORD *)a2 - 4);
        v14[0] = *(_QWORD *)(*((_QWORD *)a2 - 8) + 8LL);
        v14[1] = *(_QWORD *)(v12 + 8);
        v4 = sub_B6E160(v3, 0x22DCu, (__int64)v14, 2);
        v5 = sub_BCB2E0((_QWORD *)*v3);
        v6 = sub_ACD640(v5, 0, 0);
        v15 = 257;
        v7 = *(_QWORD *)(a1 + 304);
        v8 = 0;
        v13[0] = v6;
        v9 = *((_QWORD *)a2 - 8);
        v13[2] = v12;
        v13[1] = v9;
        v13[3] = *(_QWORD *)(v7 + 32 * (1LL - (*(_DWORD *)(v7 + 4) & 0x7FFFFFF)));
        if ( v4 )
          v8 = *(_QWORD *)(v4 + 24);
        v10 = sub_BD2C40(88, 5u);
        if ( v10 )
        {
          sub_B44260((__int64)v10, **(_QWORD **)(v8 + 16), 56, 5u, (__int64)(a2 + 24), 0);
          v10[9] = 0;
          sub_B4A290((__int64)v10, v8, v4, v13, 4, (__int64)v14, 0, 0);
        }
        sub_BD84D0((__int64)a2, (__int64)v10);
        LOBYTE(v2) = sub_B43D60(a2);
      }
    }
  }
  return v2;
}
