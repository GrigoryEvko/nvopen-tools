// Function: sub_8801C0
// Address: 0x8801c0
//
__int64 __fastcall sub_8801C0(unsigned __int8 a1, int a2, int a3, __int64 a4)
{
  _QWORD *v6; // rax
  __int64 v7; // r12
  _QWORD *v8; // rsi
  __int64 v9; // rcx
  __int64 v10; // rdx
  __int64 v11; // rax
  __int64 result; // rax
  _QWORD *v13; // rax
  __int64 v14; // rax
  _QWORD *v15; // rax
  __int64 v16; // rbx
  _QWORD *v17; // [rsp+8h] [rbp-78h]
  __m128i v18[7]; // [rsp+10h] [rbp-70h] BYREF

  sub_87A720(a1, v18, (__int64 *)&dword_4F077C8, a4);
  if ( (a1 & 0xFD) == 1 )
  {
    v6 = (_QWORD *)sub_72CBE0();
    v7 = sub_72D2E0(v6);
    v8 = sub_72BA30(byte_4F06A51[0]);
    if ( !a3 )
    {
LABEL_3:
      v9 = 0;
      v10 = 0;
      goto LABEL_4;
    }
    goto LABEL_9;
  }
  v7 = sub_72CBE0();
  v13 = (_QWORD *)sub_72CBE0();
  v14 = sub_72D2E0(v13);
  v8 = (_QWORD *)v14;
  if ( !a2 )
  {
    if ( !a3 )
      goto LABEL_3;
LABEL_9:
    v9 = 0;
    v10 = unk_4F06C60;
    goto LABEL_4;
  }
  v17 = (_QWORD *)v14;
  v15 = sub_72BA30(byte_4F06A51[0]);
  v9 = 0;
  v8 = v17;
  v10 = (__int64)v15;
  if ( a3 )
    v9 = unk_4F06C60;
LABEL_4:
  v11 = sub_732700(v7, (__int64)v8, v10, v9, 0, 0, 0, 0);
  result = sub_8800F0((__int64)v18, v11);
  if ( dword_4D048B8 )
  {
    if ( ((a1 - 2) & 0xFD) == 0 )
    {
      v16 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(result + 88) + 152LL) + 168LL);
      result = (__int64)sub_725E60();
      *(_QWORD *)(v16 + 56) = result;
    }
  }
  return result;
}
