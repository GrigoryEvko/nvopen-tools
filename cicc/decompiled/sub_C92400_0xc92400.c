// Function: sub_C92400
// Address: 0xc92400
//
void __fastcall sub_C92400(unsigned __int8 *a1, __int64 a2, __int64 a3)
{
  unsigned __int8 *v3; // r13
  unsigned __int8 *v4; // r12
  unsigned __int8 v6; // bl
  unsigned __int8 *v7; // rax
  unsigned __int64 v8; // rdx
  __int64 v9; // rdi
  __int64 v10; // rsi
  _BYTE *v11; // rax
  _BYTE *v12; // rax
  __int64 v13; // rsi
  __int64 v14; // rdi
  _BYTE *v15; // rax

  v3 = &a1[a2];
  if ( a1 != &a1[a2] )
  {
    v4 = a1;
    while ( 1 )
    {
      v6 = *v4;
      v7 = *(unsigned __int8 **)(a3 + 32);
      v8 = *(_QWORD *)(a3 + 24);
      if ( *v4 == 92 )
      {
        if ( v8 <= (unsigned __int64)v7 )
        {
          v14 = sub_CB5D20(a3, 92);
        }
        else
        {
          v14 = a3;
          *(_QWORD *)(a3 + 32) = v7 + 1;
          *v7 = 92;
        }
        v15 = *(_BYTE **)(v14 + 32);
        if ( (unsigned __int64)v15 >= *(_QWORD *)(v14 + 24) )
        {
          sub_CB5D20(v14, 92);
        }
        else
        {
          *(_QWORD *)(v14 + 32) = v15 + 1;
          *v15 = 92;
        }
        goto LABEL_6;
      }
      if ( (unsigned __int8)(v6 - 32) <= 0x5Eu && v6 != 34 )
        break;
      if ( v8 <= (unsigned __int64)v7 )
      {
        v9 = sub_CB5D20(a3, 92);
      }
      else
      {
        v9 = a3;
        *(_QWORD *)(a3 + 32) = v7 + 1;
        *v7 = 92;
      }
      v10 = (unsigned __int8)a0123456789abcd_10[v6 >> 4];
      v11 = *(_BYTE **)(v9 + 32);
      if ( (unsigned __int64)v11 >= *(_QWORD *)(v9 + 24) )
      {
        v9 = sub_CB5D20(v9, v10);
      }
      else
      {
        *(_QWORD *)(v9 + 32) = v11 + 1;
        *v11 = v10;
      }
      v12 = *(_BYTE **)(v9 + 32);
      v13 = (unsigned __int8)a0123456789abcd_10[v6 & 0xF];
      if ( (unsigned __int64)v12 >= *(_QWORD *)(v9 + 24) )
      {
        sub_CB5D20(v9, v13);
LABEL_6:
        if ( v3 == ++v4 )
          return;
      }
      else
      {
        ++v4;
        *(_QWORD *)(v9 + 32) = v12 + 1;
        *v12 = v13;
        if ( v3 == v4 )
          return;
      }
    }
    if ( v8 <= (unsigned __int64)v7 )
    {
      sub_CB5D20(a3, v6);
    }
    else
    {
      *(_QWORD *)(a3 + 32) = v7 + 1;
      *v7 = v6;
    }
    goto LABEL_6;
  }
}
