// Function: sub_33E1790
// Address: 0x33e1790
//
__int64 __fastcall sub_33E1790(__int64 a1, __int64 a2, unsigned __int8 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int16 *v7; // rax
  int v8; // edx
  __int64 v9; // rax
  __int64 result; // rax
  unsigned int v11; // eax
  unsigned __int64 v12; // rdx
  __int64 v13; // [rsp+8h] [rbp-58h]
  __int16 v14; // [rsp+20h] [rbp-40h] BYREF
  __int64 v15; // [rsp+28h] [rbp-38h]
  unsigned __int64 v16; // [rsp+30h] [rbp-30h] BYREF
  unsigned int v17; // [rsp+38h] [rbp-28h]

  v7 = (unsigned __int16 *)(*(_QWORD *)(a1 + 48) + 16LL * (unsigned int)a2);
  v8 = *v7;
  v9 = *((_QWORD *)v7 + 1);
  v14 = v8;
  v15 = v9;
  if ( (_WORD)v8 )
  {
    if ( (unsigned __int16)(v8 - 17) > 0x9Eu )
    {
LABEL_3:
      v17 = 1;
      v16 = 1;
      goto LABEL_4;
    }
    v11 = word_4456340[v8 - 1];
    v17 = v11;
    if ( v11 > 0x40 )
      goto LABEL_14;
  }
  else
  {
    if ( !sub_30070D0((__int64)&v14) )
      goto LABEL_3;
    v11 = sub_3007240((__int64)&v14);
    v17 = v11;
    if ( v11 > 0x40 )
    {
LABEL_14:
      sub_C43690((__int64)&v16, -1, 1);
      goto LABEL_4;
    }
  }
  v12 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v11;
  if ( !v11 )
    v12 = 0;
  v16 = v12;
LABEL_4:
  result = sub_33E16A0(a1, a2, (__int64)&v16, a3, a5, a6);
  if ( v17 > 0x40 )
  {
    if ( v16 )
    {
      v13 = result;
      j_j___libc_free_0_0(v16);
      return v13;
    }
  }
  return result;
}
