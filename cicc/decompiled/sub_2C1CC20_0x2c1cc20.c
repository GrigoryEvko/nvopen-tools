// Function: sub_2C1CC20
// Address: 0x2c1cc20
//
void __fastcall sub_2C1CC20(__int64 a1, __int64 a2)
{
  __int64 v3; // r14
  __int64 v4; // r15
  __int64 v5; // rax
  unsigned int **v6; // rdi
  unsigned __int8 *v7; // r13
  _BYTE *v8; // rdx
  __int64 *v9; // rax
  __int64 v10[4]; // [rsp+0h] [rbp-60h] BYREF
  __int16 v11; // [rsp+20h] [rbp-40h]

  v10[0] = *(_QWORD *)(a1 + 88);
  if ( v10[0] )
    sub_2AAAFA0(v10);
  sub_2BF1A90(a2, (__int64)v10);
  sub_9C6650(v10);
  if ( !sub_2BFB0D0(**(_QWORD **)(a1 + 48))
    || (v9 = *(__int64 **)(a1 + 48),
        BYTE4(v10[0]) = 0,
        LODWORD(v10[0]) = 0,
        (v3 = sub_2BFB120(a2, *v9, (unsigned int *)v10)) == 0) )
  {
    v3 = sub_2BFB640(a2, **(_QWORD **)(a1 + 48), 0);
  }
  v4 = sub_2BFB640(a2, *(_QWORD *)(*(_QWORD *)(a1 + 48) + 8LL), 0);
  v5 = sub_2BFB640(a2, *(_QWORD *)(*(_QWORD *)(a1 + 48) + 16LL), 0);
  v6 = *(unsigned int ***)(a2 + 904);
  v11 = 257;
  v7 = (unsigned __int8 *)sub_B36550(v6, v3, v4, v5, (__int64)v10, 0);
  sub_2BF26E0(a2, a1 + 96, (__int64)v7, 0);
  if ( (unsigned __int8)sub_920620((__int64)v7) )
    sub_2AAF930(a1, v7);
  v8 = *(_BYTE **)(a1 + 136);
  if ( v8 && *v8 <= 0x1Cu )
    v8 = 0;
  sub_2BF08A0(a2, v7, v8);
}
