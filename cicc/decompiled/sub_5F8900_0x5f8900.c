// Function: sub_5F8900
// Address: 0x5f8900
//
__int64 __fastcall sub_5F8900(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // r12
  __int64 v6; // rbx
  char v7; // al
  __int64 v8; // rdx
  __int64 v9; // rdi
  __int64 v10; // r13
  __int64 v11; // r15
  __int64 v12; // rdi
  __int64 v13; // rdx
  __int64 v14; // r14
  __int64 v16; // r15
  _QWORD *v17; // r12
  __int64 v18; // rcx
  __int64 v19; // r8
  __int64 *i; // r13
  _QWORD *v21; // rbx
  __int64 v22; // rdi
  __int64 v23; // rsi
  _QWORD *v24; // rax

  v5 = a2;
  v6 = a1;
  v7 = *(_BYTE *)(a1 + 80);
  if ( v7 == 16 )
  {
    v6 = **(_QWORD **)(a1 + 88);
    v7 = *(_BYTE *)(v6 + 80);
  }
  v8 = 0x8000000000000LL;
  if ( v7 == 24 )
    v6 = *(_QWORD *)(v6 + 88);
  v9 = *(_QWORD *)(v6 + 88);
  v10 = *(_QWORD *)(v9 + 152);
  v11 = *(_QWORD *)(v10 + 168);
  if ( (*(_QWORD *)(v9 + 200) & 0x18000001000000LL) == 0x8000000000000LL && !*(_QWORD *)(v11 + 56) )
  {
    a2 = 0;
    sub_5F8DB0(v9, 0);
  }
  v12 = v6;
  sub_894C00(v6, a2, v8, a4, a5);
  v14 = *(_QWORD *)(v11 + 56);
  if ( !v14 )
    return 1;
  if ( (*(_BYTE *)v14 & 2) != 0 )
  {
    v12 = *(_QWORD *)(v11 + 8);
    sub_5F80E0(v12);
    v14 = *(_QWORD *)(v11 + 56);
    if ( !v14 || (*(_BYTE *)v14 & 4) != 0 )
      return 1;
  }
  else if ( (*(_BYTE *)v14 & 4) != 0 )
  {
    return 1;
  }
  if ( !unk_4F06978 )
  {
    v16 = *(_QWORD *)(v5 + 168);
    v17 = *(_QWORD **)(v16 + 56);
    if ( !v17 )
    {
      v17 = (_QWORD *)sub_725E60(v12, a2, v13);
      v17[2] = *(_QWORD *)(v6 + 48);
      v17[3] = *(_QWORD *)(v6 + 48);
      *(_QWORD *)(v16 + 56) = v17;
    }
    if ( !(unsigned int)sub_8D76D0(v10) )
    {
      for ( i = *(__int64 **)(v14 + 8); i; i = (__int64 *)*i )
      {
        if ( !*((_BYTE *)i + 16) )
        {
          v21 = (_QWORD *)v17[1];
          if ( v21 )
          {
            while ( 1 )
            {
              v22 = v21[1];
              v23 = i[1];
              if ( v22 == v23 || (unsigned int)sub_8D97D0(v22, v23, 0, v18, v19) )
                break;
              v21 = (_QWORD *)*v21;
              if ( !v21 )
                goto LABEL_22;
            }
          }
          else
          {
LABEL_22:
            v24 = (_QWORD *)sub_725E90();
            v24[1] = i[1];
            *v24 = v17[1];
            v17[1] = v24;
          }
        }
      }
    }
  }
  return 0;
}
