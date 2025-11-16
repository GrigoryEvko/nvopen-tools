// Function: sub_70BAF0
// Address: 0x70baf0
//
__int64 __fastcall sub_70BAF0(unsigned __int8 a1, const __m128i *a2, _OWORD *a3, _DWORD *a4, _DWORD *a5)
{
  __int64 v7; // r9
  __int64 v8; // r12
  __int64 v9; // rdx
  __int64 v10; // r13
  char v11; // al
  __int128 *v12; // rdx
  _DWORD *v13; // r8
  __int16 v14; // ax
  __int64 result; // rax
  _DWORD *v16; // [rsp+8h] [rbp-58h] BYREF
  __int128 v17; // [rsp+10h] [rbp-50h] BYREF
  __int64 v18; // [rsp+20h] [rbp-40h]
  __int64 v19; // [rsp+28h] [rbp-38h]

  *a4 = 0;
  *a5 = 0;
  v16 = a4;
  *(_QWORD *)&v17 = sub_709B30(a1, a2);
  v8 = v17;
  v10 = v9;
  *((_QWORD *)&v17 + 1) = v9;
  if ( unk_4F07580 )
  {
    v12 = (__int128 *)((char *)&v16 + n + 7);
    v11 = *((_BYTE *)&v16 + n + 7);
  }
  else
  {
    v11 = v17;
    v12 = &v17;
  }
  v13 = v16;
  *(_BYTE *)v12 = v11 + 0x80;
  sub_709750(v17, a1, a3, v13, v7);
  v18 = v8;
  v19 = v10;
  if ( unk_4F07580 )
    v14 = *(_WORD *)((char *)&v17 + n + 14);
  else
    v14 = __ROL2__(v18, 8);
  result = v14 & 0x7FFF;
  if ( (_DWORD)result == 0x7FFF )
    *a5 = 1;
  return result;
}
