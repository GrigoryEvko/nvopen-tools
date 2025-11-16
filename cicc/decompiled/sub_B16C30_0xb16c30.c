// Function: sub_B16C30
// Address: 0xb16c30
//
__int64 __fastcall sub_B16C30(__int64 a1, _BYTE *a2, __int64 a3, __int64 a4)
{
  _QWORD *v5; // rdx
  _QWORD v7[3]; // [rsp+10h] [rbp-50h] BYREF
  __int64 v8; // [rsp+28h] [rbp-38h]
  __int64 v9; // [rsp+30h] [rbp-30h]
  __int64 v10; // [rsp+38h] [rbp-28h]
  __int64 v11; // [rsp+40h] [rbp-20h]

  *(_QWORD *)a1 = a1 + 16;
  sub_B14B30((__int64 *)a1, a2, (__int64)&a2[a3]);
  *(_QWORD *)(a1 + 32) = a1 + 48;
  *(_QWORD *)(a1 + 40) = 0;
  *(_BYTE *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 64) = 0;
  *(_QWORD *)(a1 + 72) = 0;
  v10 = 0x100000000LL;
  v7[1] = 0;
  v7[0] = &unk_49DD210;
  v7[2] = 0;
  v8 = 0;
  v9 = 0;
  v11 = a1 + 32;
  sub_CB5980(v7, 0, 0, 0);
  if ( BYTE4(a4) )
  {
    v5 = (_QWORD *)v9;
    if ( (unsigned __int64)(v8 - v9) <= 8 )
    {
      sub_CB6200(v7, "vscale x ", 9);
    }
    else
    {
      *(_BYTE *)(v9 + 8) = 32;
      *v5 = 0x7820656C61637376LL;
      v9 += 9;
    }
  }
  sub_CB59D0(v7, (unsigned int)a4);
  v7[0] = &unk_49DD210;
  return sub_CB5840(v7);
}
