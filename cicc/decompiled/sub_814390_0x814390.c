// Function: sub_814390
// Address: 0x814390
//
void __fastcall sub_814390(__int64 a1, int a2)
{
  _BYTE *v2; // r15
  __int64 v4; // rsi
  __int64 v5; // rdx
  __int64 v6; // rcx
  __int64 v7; // r8
  __int64 v8; // r9
  _QWORD *v9; // r12
  int i; // r13d
  char v11; // al
  __int64 v12; // [rsp-98h] [rbp-98h]
  __int64 v13; // [rsp-90h] [rbp-90h] BYREF
  _QWORD v14[4]; // [rsp-88h] [rbp-88h] BYREF
  char v15; // [rsp-68h] [rbp-68h]
  __int64 v16; // [rsp-60h] [rbp-60h]
  __int64 v17; // [rsp-58h] [rbp-58h]
  _BOOL4 v18; // [rsp-50h] [rbp-50h]
  char v19; // [rsp-4Ch] [rbp-4Ch]
  __int64 v20; // [rsp-48h] [rbp-48h]

  if ( (*(_BYTE *)(a1 + 89) & 8) == 0 )
  {
    v2 = (_BYTE *)a1;
    v4 = (__int64)&v13 + 4;
    if ( sub_80A070(a1, (_DWORD *)&v13 + 1) && *(char **)(a1 + 8) != off_4B6D4E0 )
    {
      v9 = 0;
      for ( i = 0; ; i = 1 )
      {
        v11 = v2[195];
        memset(&v14[1], 0, 24);
        v15 = 0;
        v16 = 0;
        v17 = 0;
        v18 = (v11 & 8) != 0;
        v19 = 0;
        v20 = 0;
        sub_809110(a1, v4, v5, v6, v7, v8, v12, v13, 0);
        sub_823800(qword_4F18BE0);
        if ( i == 1 )
          HIDWORD(v20) = 1;
        v14[0] += 2LL;
        sub_8238B0(qword_4F18BE0, &unk_3C1BC40, 2);
        if ( (unsigned __int8)(v2[174] - 1) < 2u )
          v9 = v2 + 184;
        sub_8111C0((__int64)v2, HIDWORD(v13), a2, 0, 0, v9, (__int64)v14);
        if ( (unsigned __int8)(v2[174] - 1) <= 1u )
        {
          if ( (unsigned __int8)(((v2[205] >> 2) & 7) - 1) > 3u )
            sub_721090();
          v19 = a1209[(unsigned __int8)(((v2[205] >> 2) & 7) - 1)];
        }
        v4 = 1;
        a1 = (__int64)v2;
        sub_80B290((__int64)v2, 1, (__int64)v14);
        if ( !(_DWORD)v20 || (v2[198] & 0x20) == 0 && (v2[197] & 0x60) == 0 )
          break;
        if ( i == 1 )
          break;
      }
    }
  }
}
