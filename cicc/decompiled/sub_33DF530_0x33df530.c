// Function: sub_33DF530
// Address: 0x33df530
//
__int64 __fastcall sub_33DF530(__int64 a1, __int64 a2, __int64 a3, unsigned int a4)
{
  __int64 v4; // rbx
  int v5; // eax
  unsigned __int16 *v6; // rbx
  int v7; // r12d
  int v8; // eax
  __int64 v9; // rbx
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // r8
  __int16 v16; // [rsp+0h] [rbp-50h] BYREF
  __int64 v17; // [rsp+8h] [rbp-48h]
  __int16 v18; // [rsp+10h] [rbp-40h] BYREF
  __int64 v19; // [rsp+18h] [rbp-38h]

  v4 = 16LL * (unsigned int)a3;
  v5 = sub_33D4D80(a1, a2, a3, a4);
  v6 = (unsigned __int16 *)(*(_QWORD *)(a2 + 48) + v4);
  v7 = v5;
  v8 = *v6;
  v9 = *((_QWORD *)v6 + 1);
  v16 = v8;
  v17 = v9;
  if ( (_WORD)v8 )
  {
    if ( (unsigned __int16)(v8 - 17) > 0xD3u )
    {
      v18 = v8;
      v19 = v9;
      goto LABEL_4;
    }
    LOWORD(v8) = word_4456580[v8 - 1];
    v11 = 0;
  }
  else
  {
    if ( !sub_30070B0((__int64)&v16) )
    {
      v19 = v9;
      v18 = 0;
      goto LABEL_9;
    }
    LOWORD(v8) = sub_3009970((__int64)&v16, a2, v13, v14, v15);
  }
  v18 = v8;
  v19 = v11;
  if ( !(_WORD)v8 )
  {
LABEL_9:
    LODWORD(v10) = sub_3007260((__int64)&v18);
    return (unsigned int)(v10 - v7 + 1);
  }
LABEL_4:
  if ( (_WORD)v8 == 1 || (unsigned __int16)(v8 - 504) <= 7u )
    BUG();
  v10 = *(_QWORD *)&byte_444C4A0[16 * (unsigned __int16)v8 - 16];
  return (unsigned int)(v10 - v7 + 1);
}
