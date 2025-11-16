// Function: sub_33CE690
// Address: 0x33ce690
//
__int64 __fastcall sub_33CE690(__int64 a1, __int64 a2, __int64 a3, unsigned int a4)
{
  unsigned __int16 *v6; // rax
  int v7; // edx
  __int64 v8; // rax
  unsigned __int64 v9; // rax
  unsigned int v10; // r12d
  unsigned int v12; // ebx
  unsigned int v13; // eax
  __int16 v14; // [rsp+10h] [rbp-60h] BYREF
  __int64 v15; // [rsp+18h] [rbp-58h]
  unsigned __int64 v16; // [rsp+20h] [rbp-50h] BYREF
  unsigned int v17; // [rsp+28h] [rbp-48h]
  unsigned __int64 v18; // [rsp+30h] [rbp-40h] BYREF
  unsigned int v19; // [rsp+38h] [rbp-38h]

  v6 = (unsigned __int16 *)(*(_QWORD *)(a2 + 48) + 16LL * (unsigned int)a3);
  v7 = *v6;
  v8 = *((_QWORD *)v6 + 1);
  v16 = 0;
  v17 = 1;
  v14 = v7;
  v15 = v8;
  if ( !(_WORD)v7 )
  {
    if ( sub_3007100((__int64)&v14) )
      goto LABEL_3;
    v13 = sub_3007130((__int64)&v14, a2);
    v19 = v13;
    if ( v13 <= 0x40 )
    {
LABEL_20:
      if ( v13 )
        v9 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v13;
      else
        v9 = 0;
      goto LABEL_4;
    }
LABEL_18:
    sub_C43690((__int64)&v18, -1, 1);
    goto LABEL_5;
  }
  if ( (unsigned __int16)(v7 - 176) > 0x34u )
  {
    v13 = word_4456340[v7 - 1];
    v19 = v13;
    if ( v13 <= 0x40 )
      goto LABEL_20;
    goto LABEL_18;
  }
LABEL_3:
  v19 = 1;
  v9 = 1;
LABEL_4:
  v18 = v9;
LABEL_5:
  v10 = sub_33CD9D0(a1, a2, a3, (__int64)&v18, (__int64)&v16, 0);
  if ( (_BYTE)v10 )
  {
    v10 = a4;
    if ( !(_BYTE)a4 )
    {
      v12 = v17;
      if ( v17 <= 0x40 )
        LOBYTE(v10) = v16 == 0;
      else
        LOBYTE(v10) = v12 == (unsigned int)sub_C444A0((__int64)&v16);
    }
  }
  if ( v19 > 0x40 && v18 )
    j_j___libc_free_0_0(v18);
  if ( v17 > 0x40 && v16 )
    j_j___libc_free_0_0(v16);
  return v10;
}
