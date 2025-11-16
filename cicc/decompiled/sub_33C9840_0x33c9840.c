// Function: sub_33C9840
// Address: 0x33c9840
//
bool __fastcall sub_33C9840(unsigned int *a1, __int64 a2)
{
  __int64 v2; // rbx
  unsigned __int16 *v3; // rdx
  int v4; // eax
  __int64 v5; // r12
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 v9; // rax
  unsigned int v10; // r12d
  unsigned __int64 v11; // r13
  __int64 v12; // rdx
  unsigned __int64 v13; // rax
  __int64 v15; // rdx
  __int16 v16; // [rsp-58h] [rbp-58h] BYREF
  __int64 v17; // [rsp-50h] [rbp-50h]
  __int16 v18; // [rsp-48h] [rbp-48h] BYREF
  __int64 v19; // [rsp-40h] [rbp-40h]
  __int64 v20; // [rsp-38h] [rbp-38h]
  __int64 v21; // [rsp-30h] [rbp-30h]

  if ( !*(_QWORD *)a2 )
    return 1;
  v2 = *(_QWORD *)(*(_QWORD *)a2 + 96LL);
  v3 = (unsigned __int16 *)(*(_QWORD *)(*(_QWORD *)a1 + 48LL) + 16LL * a1[2]);
  v4 = *v3;
  v5 = *((_QWORD *)v3 + 1);
  v16 = v4;
  v17 = v5;
  if ( (_WORD)v4 )
  {
    if ( (unsigned __int16)(v4 - 17) > 0xD3u )
    {
      v18 = v4;
      v19 = v5;
      goto LABEL_10;
    }
    LOWORD(v4) = word_4456580[v4 - 1];
    v15 = 0;
  }
  else
  {
    if ( !sub_30070B0((__int64)&v16) )
    {
      v19 = v5;
      v18 = 0;
      goto LABEL_5;
    }
    LOWORD(v4) = sub_3009970((__int64)&v16, a2, v6, v7, v8);
  }
  v18 = v4;
  v19 = v15;
  if ( !(_WORD)v4 )
  {
LABEL_5:
    v9 = sub_3007260((__int64)&v18);
    v10 = *(_DWORD *)(v2 + 32);
    v20 = v9;
    v11 = v9;
    v21 = v12;
    if ( v10 <= 0x40 )
    {
LABEL_6:
      v13 = *(_QWORD *)(v2 + 24);
      return v13 >= v11;
    }
    goto LABEL_13;
  }
LABEL_10:
  if ( (_WORD)v4 == 1 || (unsigned __int16)(v4 - 504) <= 7u )
    BUG();
  v10 = *(_DWORD *)(v2 + 32);
  v11 = *(_QWORD *)&byte_444C4A0[16 * (unsigned __int16)v4 - 16];
  if ( v10 <= 0x40 )
    goto LABEL_6;
LABEL_13:
  if ( v10 - (unsigned int)sub_C444A0(v2 + 24) <= 0x40 )
  {
    v13 = **(_QWORD **)(v2 + 24);
    return v13 >= v11;
  }
  return 1;
}
