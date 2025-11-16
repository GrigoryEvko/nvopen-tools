// Function: sub_3205980
// Address: 0x3205980
//
__int64 __fastcall sub_3205980(__int64 a1, unsigned __int8 *a2)
{
  __int64 v3; // rdx
  __int64 v4; // rsi
  int v5; // r9d
  unsigned int i; // eax
  __int64 v7; // rcx
  unsigned int v8; // eax
  __int64 result; // rax
  __int64 v10; // rax
  unsigned int v11; // eax
  unsigned int v12; // [rsp+Ch] [rbp-74h]
  __int16 v13; // [rsp+20h] [rbp-60h] BYREF
  int v14; // [rsp+22h] [rbp-5Eh]
  unsigned __int64 v15; // [rsp+28h] [rbp-58h]
  unsigned __int64 v16; // [rsp+30h] [rbp-50h]
  unsigned __int64 v17[2]; // [rsp+40h] [rbp-40h] BYREF
  __int64 v18; // [rsp+50h] [rbp-30h] BYREF

  v3 = *(unsigned int *)(a1 + 1240);
  v4 = *(_QWORD *)(a1 + 1224);
  if ( (_DWORD)v3 )
  {
    v5 = 1;
    for ( i = (v3 - 1) & (969526130 * (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4))); ; i = (v3 - 1) & v8 )
    {
      v7 = v4 + 24LL * i;
      if ( a2 == *(unsigned __int8 **)v7 && !*(_QWORD *)(v7 + 8) )
        break;
      if ( *(_QWORD *)v7 == -4096 && *(_QWORD *)(v7 + 8) == -4096 )
        goto LABEL_11;
      v8 = v5 + i;
      ++v5;
    }
    if ( v7 != v4 + 24 * v3 )
      return *(unsigned int *)(v7 + 16);
  }
LABEL_11:
  sub_3205740((__int64)v17, a1, a2);
  v13 = 5637;
  v15 = v17[0];
  v14 = 0;
  v16 = v17[1];
  v10 = sub_370B390(a1 + 648, &v13);
  v11 = sub_3707F80(a1 + 632, v10);
  result = sub_31FEC80(a1, (__int64)a2, v11, 0);
  if ( (__int64 *)v17[0] != &v18 )
  {
    v12 = result;
    j_j___libc_free_0(v17[0]);
    return v12;
  }
  return result;
}
