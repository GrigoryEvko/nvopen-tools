// Function: sub_1D181C0
// Address: 0x1d181c0
//
__int64 __fastcall sub_1D181C0(__int64 a1, __int64 a2, _QWORD *a3, unsigned int a4)
{
  unsigned int v4; // r12d
  __int64 *v6; // rax
  _BYTE v7[16]; // [rsp+0h] [rbp-40h] BYREF
  __int64 (__fastcall *v8)(_BYTE *, __int64, int); // [rsp+10h] [rbp-30h]
  __int64 (__fastcall *v9)(__int64, __int64 *); // [rsp+18h] [rbp-28h]

  while ( 1 )
  {
    v9 = sub_1D13E90;
    v8 = (__int64 (__fastcall *)(_BYTE *, __int64, int))sub_1D12F30;
    v4 = sub_1D169E0((_QWORD *)a2, a3, (__int64)v7, a4);
    if ( v8 )
      v8(v7, (__int64)v7, 3);
    if ( (_BYTE)v4 )
      break;
    if ( *(_WORD *)(a2 + 24) != 119 )
      return v4;
    if ( (unsigned __int8)sub_1D181C0(
                            a1,
                            *(_QWORD *)(*(_QWORD *)(a2 + 32) + 40LL),
                            *(_QWORD *)(*(_QWORD *)(a2 + 32) + 48LL)) )
      break;
    v6 = *(__int64 **)(a2 + 32);
    a2 = *v6;
    a3 = (_QWORD *)v6[1];
  }
  return 1;
}
