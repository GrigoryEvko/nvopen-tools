// Function: sub_3729DD0
// Address: 0x3729dd0
//
__int64 __fastcall sub_3729DD0(__int64 a1, __int64 a2)
{
  __int64 v2; // r14
  unsigned __int8 *v3; // r12
  __int64 v4; // rax
  __int64 v5; // r15
  __int64 v6; // r13
  __int64 v7; // rcx
  __int64 v8; // r15
  __int64 v9; // r14
  __int64 result; // rax
  unsigned __int8 *v11; // rax
  __int64 v12; // r14
  __int64 v13; // rdx
  __int64 v14; // rcx
  unsigned __int64 v15; // r8
  __int64 v16; // r9
  __int64 v17; // [rsp+0h] [rbp-40h]
  char v18; // [rsp+Fh] [rbp-31h]

  v2 = a2;
  v3 = 0;
  v4 = *(_QWORD *)(a1 + 8);
  v5 = *(_QWORD *)a2;
  v6 = *(_QWORD *)(*(_QWORD *)(v4 + 224) + 16LL);
  if ( (*(_BYTE *)(*(_QWORD *)a2 + 2LL) & 8) != 0 )
  {
    v11 = (unsigned __int8 *)sub_B2E500(*(_QWORD *)a2);
    v3 = sub_BD3990(v11, a2);
    if ( *v3 )
      v3 = 0;
    if ( (*(_BYTE *)(v5 + 2) & 8) != 0 && !(unsigned int)sub_B2A630((__int64)v3) )
    {
      if ( (unsigned int)sub_A746B0((_QWORD *)(v5 + 120))
        || (a2 = 41, !(unsigned __int8)sub_B2D610(v5, 41))
        || (*(_BYTE *)(v5 + 2) & 8) != 0 )
      {
        v12 = **(_QWORD **)(*(_QWORD *)(a1 + 8) + 232LL);
        if ( (unsigned int)sub_A746B0((_QWORD *)(v12 + 120)) )
          goto LABEL_12;
        a2 = 41;
        v18 = sub_B2D610(v12, 41);
        if ( !v18 || (*(_BYTE *)(v12 + 2) & 8) != 0 )
          goto LABEL_12;
        goto LABEL_18;
      }
    }
    v4 = *(_QWORD *)(a1 + 8);
  }
  v7 = *(_QWORD *)(v2 + 432);
  v8 = *(_QWORD *)(v2 + 440);
  v9 = **(_QWORD **)(v4 + 232);
  v17 = v7;
  v18 = v8 != v7;
  if ( (unsigned int)sub_A746B0((_QWORD *)(v9 + 120))
    || (a2 = 41, !(unsigned __int8)sub_B2D610(v9, 41))
    || (*(_BYTE *)(v9 + 2) & 8) != 0 )
  {
    if ( v8 == v17 )
      goto LABEL_4;
    goto LABEL_12;
  }
LABEL_18:
  if ( !v18 )
  {
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v6 + 104LL))(v6);
    goto LABEL_4;
  }
LABEL_12:
  if ( v3 )
  {
    a2 = sub_31DB510(*(_QWORD *)(a1 + 8), (__int64)v3);
    (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)v6 + 112LL))(v6, a2);
  }
  (*(void (__fastcall **)(__int64))(*(_QWORD *)v6 + 128LL))(v6);
  sub_32530C0((char *)a1, a2, v13, v14, v15, v16);
LABEL_4:
  result = *(_QWORD *)(*(_QWORD *)(a1 + 8) + 208LL);
  if ( *(_DWORD *)(result + 336) == 3 )
    return (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v6 + 96LL))(v6);
  return result;
}
