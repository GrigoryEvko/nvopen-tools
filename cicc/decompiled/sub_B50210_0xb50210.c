// Function: sub_B50210
// Address: 0xb50210
//
void __fastcall sub_B50210(__int64 a1, int a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6, __int64 a7, __int64 a8)
{
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // rcx
  __int64 v13; // rsi
  __int64 v14; // rax
  bool v15; // zf
  __int64 v16; // rdx
  __int64 v17; // rdx

  sub_B44260(a1, a4, a2, 1u, a7, a8);
  if ( *(_QWORD *)(a1 - 32) )
  {
    v10 = *(_QWORD *)(a1 - 24);
    **(_QWORD **)(a1 - 16) = v10;
    if ( v10 )
      *(_QWORD *)(v10 + 16) = *(_QWORD *)(a1 - 16);
  }
  *(_QWORD *)(a1 - 32) = a3;
  if ( a3 )
  {
    v11 = *(_QWORD *)(a3 + 16);
    v12 = a3 + 16;
    *(_QWORD *)(a1 - 24) = v11;
    if ( v11 )
    {
      v13 = a1 - 24;
      *(_QWORD *)(v11 + 16) = a1 - 24;
      v14 = a1 - 32;
      v15 = *(_QWORD *)(a1 - 32) == 0;
      *(_QWORD *)(a1 - 16) = v12;
      *(_QWORD *)(a3 + 16) = a1 - 32;
      if ( v15 )
      {
        *(_QWORD *)(a1 - 32) = a3;
        v17 = a1 - 32;
        *(_QWORD *)(a1 - 24) = v14;
        goto LABEL_10;
      }
      v16 = *(_QWORD *)(a1 - 24);
      *(_QWORD *)(a3 + 16) = v16;
      if ( v16 )
      {
LABEL_8:
        *(_QWORD *)(v16 + 16) = *(_QWORD *)(a1 - 16);
        v17 = *(_QWORD *)(a3 + 16);
        *(_QWORD *)(a1 - 32) = a3;
        *(_QWORD *)(a1 - 24) = v17;
        if ( !v17 )
        {
LABEL_11:
          *(_QWORD *)(a1 - 16) = v12;
          *(_QWORD *)(a3 + 16) = v14;
          goto LABEL_12;
        }
        v13 = a1 - 24;
LABEL_10:
        *(_QWORD *)(v17 + 16) = v13;
        goto LABEL_11;
      }
    }
    else
    {
      v16 = *(_QWORD *)(a1 - 24);
      v14 = a1 - 32;
      *(_QWORD *)(a1 - 16) = v12;
      *(_QWORD *)(a3 + 16) = a1 - 32;
      *(_QWORD *)(a3 + 16) = v16;
      if ( v16 )
        goto LABEL_8;
    }
    *(_QWORD *)(a1 - 32) = a3;
    *(_QWORD *)(a1 - 24) = 0;
    goto LABEL_11;
  }
LABEL_12:
  sub_BD6B50(a1, a5);
  nullsub_66();
}
