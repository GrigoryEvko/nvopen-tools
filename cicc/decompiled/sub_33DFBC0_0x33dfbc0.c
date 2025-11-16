// Function: sub_33DFBC0
// Address: 0x33dfbc0
//
__int64 __fastcall sub_33DFBC0(
        __int64 a1,
        unsigned int a2,
        unsigned __int8 a3,
        unsigned __int8 a4,
        __int64 a5,
        __int64 a6)
{
  unsigned __int16 *v8; // rax
  int v9; // edx
  __int64 v10; // rax
  __int64 result; // rax
  unsigned int v12; // eax
  unsigned __int64 v13; // rdx
  __int64 v14; // [rsp+8h] [rbp-68h]
  __int16 v15; // [rsp+20h] [rbp-50h] BYREF
  __int64 v16; // [rsp+28h] [rbp-48h]
  unsigned __int64 v17; // [rsp+30h] [rbp-40h] BYREF
  unsigned int v18; // [rsp+38h] [rbp-38h]

  v8 = (unsigned __int16 *)(*(_QWORD *)(a1 + 48) + 16LL * a2);
  v9 = *v8;
  v10 = *((_QWORD *)v8 + 1);
  v15 = v9;
  v16 = v10;
  if ( (_WORD)v9 )
  {
    if ( (unsigned __int16)(v9 - 17) > 0x9Eu )
    {
LABEL_3:
      v18 = 1;
      v17 = 1;
      goto LABEL_4;
    }
    v12 = word_4456340[v9 - 1];
    v18 = v12;
    if ( v12 > 0x40 )
      goto LABEL_14;
  }
  else
  {
    if ( !sub_30070D0((__int64)&v15) )
      goto LABEL_3;
    v12 = sub_3007240((__int64)&v15);
    v18 = v12;
    if ( v12 > 0x40 )
    {
LABEL_14:
      sub_C43690((__int64)&v17, -1, 1);
      goto LABEL_4;
    }
  }
  v13 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v12;
  if ( !v12 )
    v13 = 0;
  v17 = v13;
LABEL_4:
  result = sub_33D2320(a1, a2, (__int64)&v17, a3, a4, a6);
  if ( v18 > 0x40 )
  {
    if ( v17 )
    {
      v14 = result;
      j_j___libc_free_0_0(v17);
      return v14;
    }
  }
  return result;
}
