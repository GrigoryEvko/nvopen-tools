// Function: sub_33DFAD0
// Address: 0x33dfad0
//
__int64 __fastcall sub_33DFAD0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, unsigned int a5)
{
  __int64 v5; // rbx
  int v6; // eax
  unsigned __int16 *v7; // rbx
  int v8; // r12d
  int v9; // eax
  __int64 v10; // rbx
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // r8
  __int16 v17; // [rsp+0h] [rbp-50h] BYREF
  __int64 v18; // [rsp+8h] [rbp-48h]
  __int16 v19; // [rsp+10h] [rbp-40h] BYREF
  __int64 v20; // [rsp+18h] [rbp-38h]

  v5 = 16LL * (unsigned int)a3;
  v6 = sub_33D25A0(a1, a2, a3, a4, a5);
  v7 = (unsigned __int16 *)(*(_QWORD *)(a2 + 48) + v5);
  v8 = v6;
  v9 = *v7;
  v10 = *((_QWORD *)v7 + 1);
  v17 = v9;
  v18 = v10;
  if ( (_WORD)v9 )
  {
    if ( (unsigned __int16)(v9 - 17) > 0xD3u )
    {
      v19 = v9;
      v20 = v10;
      goto LABEL_4;
    }
    LOWORD(v9) = word_4456580[v9 - 1];
    v12 = 0;
  }
  else
  {
    if ( !sub_30070B0((__int64)&v17) )
    {
      v20 = v10;
      v19 = 0;
      goto LABEL_9;
    }
    LOWORD(v9) = sub_3009970((__int64)&v17, a2, v14, v15, v16);
  }
  v19 = v9;
  v20 = v12;
  if ( !(_WORD)v9 )
  {
LABEL_9:
    LODWORD(v11) = sub_3007260((__int64)&v19);
    return (unsigned int)(v11 - v8 + 1);
  }
LABEL_4:
  if ( (_WORD)v9 == 1 || (unsigned __int16)(v9 - 504) <= 7u )
    BUG();
  v11 = *(_QWORD *)&byte_444C4A0[16 * (unsigned __int16)v9 - 16];
  return (unsigned int)(v11 - v8 + 1);
}
