// Function: sub_38480C0
// Address: 0x38480c0
//
__int64 __fastcall sub_38480C0(__int64 a1, __int64 a2)
{
  unsigned __int64 *v2; // rax
  __int64 v3; // r13
  unsigned __int64 v4; // r14
  __int64 v5; // rax
  __int16 v6; // dx
  __int64 v7; // rax
  __int64 v8; // rdx
  _QWORD *v9; // rax
  __int64 v11; // [rsp+0h] [rbp-50h] BYREF
  __int64 v12; // [rsp+8h] [rbp-48h]
  __int64 v13; // [rsp+10h] [rbp-40h] BYREF
  __int64 v14; // [rsp+18h] [rbp-38h]
  __int16 v15; // [rsp+20h] [rbp-30h] BYREF
  __int64 v16; // [rsp+28h] [rbp-28h]

  v2 = *(unsigned __int64 **)(a2 + 40);
  LODWORD(v12) = 0;
  LODWORD(v14) = 0;
  v11 = 0;
  v3 = v2[1];
  v13 = 0;
  v4 = *v2;
  v5 = *(_QWORD *)(*v2 + 48) + 16LL * *((unsigned int *)v2 + 2);
  v6 = *(_WORD *)v5;
  v7 = *(_QWORD *)(v5 + 8);
  v15 = v6;
  v16 = v7;
  if ( v6 )
  {
    if ( (unsigned __int16)(v6 - 2) <= 7u
      || (unsigned __int16)(v6 - 17) <= 0x6Cu
      || (unsigned __int16)(v6 - 176) <= 0x1Fu )
    {
      goto LABEL_5;
    }
  }
  else if ( sub_3007070((__int64)&v15) )
  {
LABEL_5:
    sub_375E510(a1, v4, v3, (__int64)&v11, (__int64)&v13);
    goto LABEL_6;
  }
  sub_375E6F0(a1, v4, v3, (__int64)&v11, (__int64)&v13);
LABEL_6:
  v8 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a2 + 40) + 40LL) + 96LL);
  v9 = *(_QWORD **)(v8 + 24);
  if ( *(_DWORD *)(v8 + 32) > 0x40u )
    v9 = (_QWORD *)*v9;
  if ( v9 )
    return v13;
  else
    return v11;
}
