// Function: sub_3846070
// Address: 0x3846070
//
void __fastcall sub_3846070(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __m128i a5)
{
  unsigned __int64 *v7; // rax
  unsigned __int64 v8; // r15
  __int64 v9; // rcx
  __int64 v10; // rax
  __int16 v11; // dx
  __int64 v12; // rax
  __int64 v13; // rdx
  _QWORD *v14; // rax
  bool v15; // zf
  __int64 *v16; // rax
  bool v17; // al
  __int64 v18; // [rsp+8h] [rbp-48h]
  __int16 v19; // [rsp+10h] [rbp-40h] BYREF
  __int64 v20; // [rsp+18h] [rbp-38h]

  v7 = *(unsigned __int64 **)(a2 + 40);
  v8 = *v7;
  v9 = v7[1];
  v10 = *(_QWORD *)(*v7 + 48) + 16LL * *((unsigned int *)v7 + 2);
  v11 = *(_WORD *)v10;
  v12 = *(_QWORD *)(v10 + 8);
  v19 = v11;
  v20 = v12;
  if ( v11 )
  {
    if ( (unsigned __int16)(v11 - 2) <= 7u
      || (unsigned __int16)(v11 - 17) <= 0x6Cu
      || (unsigned __int16)(v11 - 176) <= 0x1Fu )
    {
      goto LABEL_5;
    }
  }
  else
  {
    v18 = v9;
    v17 = sub_3007070((__int64)&v19);
    v9 = v18;
    if ( v17 )
    {
LABEL_5:
      sub_375E510((__int64)a1, v8, v9, a3, a4);
      goto LABEL_6;
    }
  }
  sub_375E6F0((__int64)a1, v8, v9, a3, a4);
LABEL_6:
  v13 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a2 + 40) + 40LL) + 96LL);
  v14 = *(_QWORD **)(v13 + 24);
  if ( *(_DWORD *)(v13 + 32) > 0x40u )
    v14 = (_QWORD *)*v14;
  v15 = v14 == 0;
  v16 = (__int64 *)a3;
  if ( !v15 )
    v16 = (__int64 *)a4;
  sub_375AEA0(a1, *v16, v16[1], a3, a4, a5);
}
